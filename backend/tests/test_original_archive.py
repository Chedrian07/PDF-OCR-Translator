"""원문 ZIP 서비스의 저장·호환·완전성 계약."""
import io
import json
import re
import zipfile

import pytest

from app.jobs import JobStore
from app.pipeline import artifacts
from conftest import wait_done


def upload(client, sample_pdf, **data):
    return client.post('/api/jobs', files={'file': ('paper.pdf', sample_pdf, 'application/pdf')}, data=data)


def test_workflow_persists_and_defaults(client, sample_pdf):
    jid = upload(client, sample_pdf, workflow='markdown_zip').json()['job_id']
    body = wait_done(client, jid)
    assert body['workflow'] == 'markdown_zip'
    assert body['result']['original_archive_url'].endswith('/archive?scope=original')
    listing = client.get('/api/jobs').json()['jobs']
    assert next(j for j in listing if j['job_id'] == jid)['workflow'] == 'markdown_zip'
    job = client.app.state.store.get(jid)
    restored = JobStore(job.dir.parent)
    restored.load_existing()
    assert restored.get(jid).workflow == 'markdown_zip'
    meta = artifacts.meta(job.dir)
    old = json.loads(meta.read_text())
    del old['workflow']
    meta.write_text(json.dumps(old))
    restored.load_existing()
    assert restored.get(jid).workflow == 'reader'
    legacy_jid = upload(client, sample_pdf).json()['job_id']
    assert wait_done(client, legacy_jid)['workflow'] == 'reader'
    assert upload(client, sample_pdf, workflow='unknown').status_code == 400


def test_original_archive_contents_and_independent_cache(client, sample_pdf):
    jid = upload(client, sample_pdf, workflow='markdown_zip').json()['job_id']
    done = wait_done(client, jid)
    job = client.app.state.store.get(jid)
    url = done['result']['original_archive_url']
    response = client.get(url)
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        names = set(zf.namelist())
        assert {f'pages/page_{i:04d}.png' for i in range(1, 4)} <= names
        assert 'result.md' in names
        assert 'images/p0001_0.jpg' in names
        assert 'meta.json' not in names and 'images/boxes.json' not in names
        assert all(n == 'result.md' or n.startswith(('images/', 'pages/')) for n in names)
        refs = re.findall(r'!\[[^\]]*\]\((images/[^)]+)\)', zf.read('result.md').decode())
        assert refs and all(ref in names for ref in refs)
    cache = artifacts.archive(job.dir, 'original')
    stamp = cache.stat().st_mtime_ns
    (job.dir / 'result.ko.md').write_text('번역')
    assert client.get(url).content == response.content
    assert cache.stat().st_mtime_ns == stamp
    with zipfile.ZipFile(io.BytesIO(client.get(f'/api/jobs/{jid}/archive').content)) as zf:
        assert 'result.ko.md' in zf.namelist()
        assert 'pages/page_0001.png' not in zf.namelist()
    events = client.get(f'/api/jobs/{jid}/events').text
    assert 'original_archive_url' in events
    assert client.get(f'/api/jobs/{jid}/archive?scope=invalid').status_code == 400


@pytest.mark.parametrize('missing', ['result.md', 'pages/page_0002.png', 'images/p0001_0.jpg'])
def test_original_archive_refuses_missing_artifacts_even_when_cached(client, sample_pdf, missing):
    jid = upload(client, sample_pdf).json()['job_id']
    wait_done(client, jid)
    url = f'/api/jobs/{jid}/archive?scope=original'
    assert client.get(url).status_code == 200
    job = client.app.state.store.get(jid)
    (job.dir / missing).unlink()
    assert client.get(url).status_code == 409


def test_original_archive_missing_during_build_is_not_cached(client, sample_pdf, monkeypatch):
    jid = upload(client, sample_pdf).json()['job_id']
    wait_done(client, jid)
    job = client.app.state.store.get(jid)
    real_write = zipfile.ZipFile.write

    def remove_page(zf, filename, arcname=None, *args, **kwargs):
        if arcname == 'pages/page_0002.png':
            filename.unlink()
        return real_write(zf, filename, arcname, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, 'write', remove_page)
    assert client.get(f'/api/jobs/{jid}/archive?scope=original').status_code == 409
    assert not artifacts.archive(job.dir, 'original').exists()
    assert not list(job.dir.glob('.archive.original.*.tmp'))


def test_original_archive_concurrent_requests(client, sample_pdf):
    from concurrent.futures import ThreadPoolExecutor

    jid = upload(client, sample_pdf).json()['job_id']
    wait_done(client, jid)
    url = f'/api/jobs/{jid}/archive?scope=original'
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: client.get(url), range(3)))
    for response in results:
        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            assert zf.testzip() is None
            assert 'pages/page_0003.png' in zf.namelist()
    job = client.app.state.store.get(jid)
    assert not list(job.dir.glob('.archive.original.*.tmp'))
