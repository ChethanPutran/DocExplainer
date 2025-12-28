## 1️⃣ Install missing XCB dependencies (Linux)

For **Ubuntu/Debian**:

```bash
sudo apt update
sudo apt install --no-install-recommends \
    libxcb-xinerama0 \
    libxcb-xinerama0-dev \
    libxcb-icccm4 \
    libxcb-icccm4-dev \
    libxcb-image0 \
    libxcb-image0-dev \
    libxcb-keysyms1 \
    libxcb-keysyms1-dev \
    libxcb-render-util0 \
    libxcb-render-util0-dev \
    libxcb-shape0 \
    libxcb-shape0-dev \
    libxcb-randr0 \
    libxcb-randr0-dev \
    libxcb-cursor-dev \
    libxkbcommon-x11-0
```

* `libxcb-cursor0` or `libxcb-cursor-dev` is the **critical package** mentioned in your error.
* These packages provide Qt the ability to interface with the X11 display.

---

## 2️⃣ Optional: Install other Qt XCB dependencies

```bash
sudo apt install libxcb-xfixes0 libxrender1 libxrandr2 libxi6 libx11-xcb1 libxkbcommon-x11-0
```

---

## 3️⃣ Verify installation

After installing, try running your app again:

```bash
python -m gui.main_window
```

It should launch without the “**Could not load the Qt platform plugin 'xcb'**” error.

---

## 4️⃣ Notes

* This is **only needed on Linux**.
* On **Windows** or **macOS**, PySide6 bundles all required platform plugins automatically.
* If you are running in **WSL2**, you also need an **X server** like `VcXsrv` or `X410` and export the display:

```bash
export DISPLAY=:0
```

---
sudo apt-get install portaudio19-dev python3-pyaudio
