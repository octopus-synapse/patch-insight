# Releases and reproducible publication

Stable releases start with a version tag matching pyproject.toml, e.g. v0.2.0.
Ordinary pushes run CI only. Commits use English Conventional Commits without
Co-authored-by trailers. No GPU training runs on GitHub-hosted runners.

Targets: GitHub Releases (wheel, source distribution, curated sources, SHA256SUMS,
model/data revision manifest, image digests, test evidence), GHCR (version, code
SHA, latest after successful publication) and the existing Hugging Face repos.
Model weights and software versions are separate. The 0.2 release keeps the
0.1 checkpoint and adds evaluation data/cards; historical revisions remain fixed.

## One-time setup

GitHub Actions uses its short-lived GITHUB_TOKEN with contents:write and
packages:write. Hugging Face needs a repository secret named HF_TOKEN with
fine-grained write access only to enzoferracini/patchinsight-bertimbau-ptbr and
enzoferracini/patchinsight-synthetic-ptbr. Never commit or print the token.
The local OAuth session is temporary and is not installed as a CI secret.

The v0.2.0 package is public and anonymous manifest access was verified.
For other first GHCR publications the documented default is private. An organization package admin
must make it public once under the package Settings → Change visibility:
https://github.com/orgs/octopus-synapse/packages/container/patch-insight/settings
The available GitHub REST/GraphQL APIs do not expose a visibility mutation.
After changing it, verify an anonymous pull; subsequent image tags reuse the
package visibility. Reference: https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry

## Publish and recover

Run CI, update the package version/lock and pinned release configuration, commit,
and push an annotated vX.Y.Z tag. The workflow downloads a fixed Hub revision,
validates its manifest, runs real-model tests, builds packages, tests Docker
without network, publishes Hub tags and image tags, then publishes the release.
The latest image advances last. Actions are pinned by commit SHA.

Use workflow_dispatch with release_tag to resume an existing version. Checkout
uses the selected tag, not the workflow branch. Existing Hub tags must contain
the exact code commit and are reused without upload; this allows recovery of a
release already published locally without copying an OAuth token into secrets.
New Hub tags require HF_TOKEN. Publishing is serialized and cancelled runs do
not move completed historical Hub tags. Partial failures remain visible in Actions.

Artifacts live in allowlisted staging directories; PDF, credentials and the
private planning file docs.md are excluded. Logs and experimental weights under
artifacts are local unless explicitly selected for a release.
