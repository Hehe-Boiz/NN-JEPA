const sourceSelect = document.getElementById("sourceSelect");
const sessionSelect = document.getElementById("sessionSelect");
const reloadButton = document.getElementById("reloadButton");
const playPauseButton = document.getElementById("playPauseButton");
const prevFrameButton = document.getElementById("prevFrameButton");
const nextFrameButton = document.getElementById("nextFrameButton");
const fpsInput = document.getElementById("fpsInput");
const frameSlider = document.getElementById("frameSlider");
const frameImage = document.getElementById("frameImage");
const frameLabel = document.getElementById("frameLabel");
const sessionLabel = document.getElementById("sessionLabel");
const frameCount = document.getElementById("frameCount");
const fileList = document.getElementById("fileList");

const state = {
  source: "raw",
  sessions: [],
  currentSession: null,
  currentIndex: 0,
  playing: false,
  timerId: null,
};

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

function stopPlayback() {
  if (state.timerId !== null) {
    clearInterval(state.timerId);
    state.timerId = null;
  }
  state.playing = false;
  playPauseButton.textContent = "Play";
}

function startPlayback() {
  stopPlayback();
  const fps = Math.max(1, Number.parseInt(fpsInput.value, 10) || 12);
  state.playing = true;
  playPauseButton.textContent = "Pause";
  state.timerId = window.setInterval(() => {
    if (!state.currentSession) {
      stopPlayback();
      return;
    }
    if (state.currentIndex >= state.currentSession.frames.length - 1) {
      stopPlayback();
      return;
    }
    setFrame(state.currentIndex + 1);
  }, Math.round(1000 / fps));
}

function togglePlayback() {
  if (state.playing) {
    stopPlayback();
  } else {
    startPlayback();
  }
}

function mediaUrl(sessionId, frameName) {
  return `/media/${state.source}/${encodeURIComponent(sessionId)}/${encodeURIComponent(frameName)}`;
}

function updateMeta() {
  if (!state.currentSession) {
    frameCount.textContent = "-";
    fileList.textContent = "-";
    frameLabel.textContent = "Frame 0 / 0";
    sessionLabel.textContent = "No session loaded";
    return;
  }
  frameCount.textContent = String(state.currentSession.frame_count);
  fileList.textContent = state.currentSession.files.join(", ");
  frameLabel.textContent = `Frame ${state.currentIndex + 1} / ${state.currentSession.frames.length}`;
  sessionLabel.textContent = `${state.currentSession.session_id} (${state.currentSession.source})`;
}

function setFrame(index) {
  if (!state.currentSession) {
    return;
  }
  const clamped = Math.max(0, Math.min(index, state.currentSession.frames.length - 1));
  state.currentIndex = clamped;
  frameSlider.value = String(clamped);
  const frameName = state.currentSession.frames[clamped];
  frameImage.src = mediaUrl(state.currentSession.session_id, frameName);
  updateMeta();

  const preloadIndex = clamped + 1;
  if (preloadIndex < state.currentSession.frames.length) {
    const preloadImage = new Image();
    preloadImage.src = mediaUrl(state.currentSession.session_id, state.currentSession.frames[preloadIndex]);
  }
}

async function loadSession(sessionId) {
  stopPlayback();
  const payload = await fetchJson(`/api/session?source=${encodeURIComponent(state.source)}&session_id=${encodeURIComponent(sessionId)}`);
  state.currentSession = payload;
  state.currentIndex = 0;
  frameSlider.min = "0";
  frameSlider.max = String(Math.max(0, payload.frames.length - 1));
  frameSlider.value = "0";
  setFrame(0);
}

async function loadSessions() {
  stopPlayback();
  state.source = sourceSelect.value;
  const payload = await fetchJson(`/api/sessions?source=${encodeURIComponent(state.source)}`);
  state.sessions = payload.sessions;

  sessionSelect.innerHTML = "";
  for (const session of state.sessions) {
    const option = document.createElement("option");
    option.value = session.session_id;
    option.textContent = `${session.session_id} (${session.frame_count})`;
    sessionSelect.appendChild(option);
  }

  if (state.sessions.length === 0) {
    state.currentSession = null;
    frameImage.removeAttribute("src");
    updateMeta();
    return;
  }

  await loadSession(state.sessions[0].session_id);
}

sourceSelect.addEventListener("change", () => {
  loadSessions().catch(showError);
});

reloadButton.addEventListener("click", () => {
  loadSessions().catch(showError);
});

sessionSelect.addEventListener("change", () => {
  loadSession(sessionSelect.value).catch(showError);
});

playPauseButton.addEventListener("click", togglePlayback);
prevFrameButton.addEventListener("click", () => {
  stopPlayback();
  setFrame(state.currentIndex - 1);
});
nextFrameButton.addEventListener("click", () => {
  stopPlayback();
  setFrame(state.currentIndex + 1);
});
frameSlider.addEventListener("input", () => {
  stopPlayback();
  setFrame(Number.parseInt(frameSlider.value, 10) || 0);
});
fpsInput.addEventListener("change", () => {
  if (state.playing) {
    startPlayback();
  }
});

window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) {
    return;
  }
  if (event.code === "Space") {
    event.preventDefault();
    togglePlayback();
  } else if (event.code === "ArrowRight") {
    event.preventDefault();
    stopPlayback();
    setFrame(state.currentIndex + 1);
  } else if (event.code === "ArrowLeft") {
    event.preventDefault();
    stopPlayback();
    setFrame(state.currentIndex - 1);
  }
});

function showError(error) {
  console.error(error);
  stopPlayback();
  sessionLabel.textContent = `Error: ${error.message}`;
}

loadSessions().catch(showError);
