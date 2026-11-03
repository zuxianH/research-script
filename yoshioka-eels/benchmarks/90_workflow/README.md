# Shared workflow and resume tools

Workflow infrastructure.

[All benchmark groups](../README.md)

| Script | Purpose |
|---|---|
| [guarded_run.py](guarded_run.py) | Reproduction entry point with transitive source/constants cache guards. |
| [repair_cold_workspace.py](repair_cold_workspace.py) | Prepare fresh repair outputs while reusing immutable reference inputs. |
| [repair_resume.py](repair_resume.py) | Resume a selected repair task under the audited current provenance profile. |
| [workflow_io.py](workflow_io.py) | Small provenance/checkpoint helpers for the material-to-EELS benchmarks. |

Use the existing `python -m benchmarks.<module>` commands from the project root. The grouped file path is for browsing and editing.
