"""
Downloads GitHub Actions artifacts (zip files containing JUnit XML reports)
and extracts them to a local cache folder, ready for parsing.

GitHub stores artifacts as zips behind a redirect URL. This module handles
the download + extraction; ingester/test_results.py handles parsing the
extracted XML files.

IMPORTANT: unlike most read endpoints, the artifact download endpoint
requires a valid GITHUB_TOKEN even for fully public repos -- you will get
a 401 Unauthorized without one, regardless of rate limit status. Make sure
GITHUB_TOKEN is set before running build_dataset.py for real.
"""
import os
import zipfile
import requests
from . import config

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "artifacts")


def get_artifact_size_bytes(repo: str, artifact_id: int) -> int:
    """Look up an artifact's declared size without downloading it."""
    url = f"{config.GITHUB_API_BASE}/repos/{repo}/actions/artifacts/{artifact_id}"
    resp = requests.get(url, headers=config.REQUEST_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json().get("size_in_bytes", 0)


def download_artifact(repo: str, artifact_id: int, dest_dir: str = None,
                       max_size_bytes: int = 50 * 1024 * 1024, timeout_seconds: int = 60) -> str:
    """
    Download a single artifact zip by ID and extract it.
    Returns the path to the extracted folder.

    Skips (raises) artifacts larger than max_size_bytes (default 50MB) --
    some CI artifacts (esp. on large projects like Kafka) can be hundreds of
    MB, which makes downloading every single one impractical for a quick
    iteration loop. Raise this limit later if you want full coverage.
    """
    size = get_artifact_size_bytes(repo, artifact_id)
    if size > max_size_bytes:
        raise ValueError(f"artifact {artifact_id} is {size / 1e6:.1f}MB, "
                          f"exceeds max_size_bytes ({max_size_bytes / 1e6:.0f}MB) -- skipping")

    dest_dir = dest_dir or CACHE_DIR
    extract_path = os.path.join(dest_dir, repo.replace("/", "_"), str(artifact_id))
    os.makedirs(extract_path, exist_ok=True)

    url = f"{config.GITHUB_API_BASE}/repos/{repo}/actions/artifacts/{artifact_id}/zip"
    resp = requests.get(url, headers=config.REQUEST_HEADERS, timeout=timeout_seconds, stream=True)
    resp.raise_for_status()

    zip_path = os.path.join(extract_path, "artifact.zip")
    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded > max_size_bytes:
                f.close()
                os.remove(zip_path)
                raise ValueError(f"artifact {artifact_id} exceeded {max_size_bytes / 1e6:.0f}MB "
                                  f"mid-download -- aborted")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_path)
    os.remove(zip_path)  # keep only the extracted contents

    return extract_path


def find_junit_xml_files(folder: str) -> list[str]:
    """Walk an extracted artifact folder and return paths to all XML files."""
    xml_files = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.endswith(".xml"):
                xml_files.append(os.path.join(root, name))
    return xml_files


if __name__ == "__main__":
    # This needs a real artifact ID from a real run, which we don't have
    # without GITHUB_TOKEN + a repo that actually has test-report artifacts
    # attached. octocat/Hello-World (used in earlier smoke tests) has none.
    # Run build_dataset.py instead, which discovers real artifact IDs first.
    print("Run 'python -m ingester.build_dataset' instead -- it discovers "
          "real artifact IDs from your configured TARGET_REPOS before "
          "calling download_artifact() on each one.")
