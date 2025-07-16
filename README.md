# 🎥 Story Dumps - Processing Backend (Backend1)

This is the **processing backend** of the [Story Dumps](https://www.storydumps.xyz) project — responsible for AI voice generation, video composition, subtitle syncing, and stitching gameplay + character animations into a final brain-rot video.

> ⚡ Built with **FastAPI**, **Kokoro TTS**, and **FFmpeg**

---

## 🔗 Related Projects

- 🎬 **Frontend**: [https://www.storydumps.xyz](https://www.storydumps.xyz)
- 🧠 **Main Backend (Django)**: [GitHub - storydumps-backend](https://github.com/manideepanasuri/Story-dumps-backend)
- 🌐 **Frontend Repo**: [GitHub - storydumps-frontend](https://github.com/manideepanasuri/Story-Dumps)

---

## ⚙️ Responsibilities

- 🧠 Accepts script and metadata from Django backend
- 🎙️ Generates character-wise dialogue audio using **Kokoro TTS**
- 🕹️ Stitches random gameplay background
- 📄 Syncs subtitles with audio timing
- 🎞️ Uses **FFmpeg** to compose the final video
- 📤 Uploads completed video to storage / returns it to main backend

---

## 📦 Tech Stack

- **Framework**: FastAPI
- **TTS Engine**: Kokoro TTS (custom voice model support)
- **Video Processing**: FFmpeg
- **Subtitles**: ASS/SSA subtitle overlays
- **Storage**: Local / MinIO-compatible
- **Concurrency**: `asyncio` + `concurrent.futures` for heavy tasks
