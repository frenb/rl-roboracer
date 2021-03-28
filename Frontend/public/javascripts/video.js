import { VideoPlayer } from "./video-player.js";
import { registerGamepadEvents, registerKeyboardEvents, registerMouseEvents, sendClickEvent } from "./register-events.js";

var videoStreamer;
var mainPlayerElement;
var mainVideoElement;
var mainThumbnailElement;
var extraVideoElements = [];
var elementObservers = [];


window.document.oncontextmenu = function () {
  return false;     // cancel default menu
}

/* Might have a use for this in the future*/
window.addEventListener('resize', function() {
  if (videoStreamer) {
    videoStreamer.resizeVideo();
  }
}, true);



function setMainVideoPlayer(playerElementId, mainTrackIndex, thumbnailTrackIndex) {
  mainPlayerElement = document.getElementById(playerElementId);

  // add video player
  mainVideoElement = document.createElement('video');
  mainVideoElement.id = videoElementId(playerElementId);
  mainVideoElement.className = "StreamVideo";
  mainVideoElement.style.touchAction = 'none';
  mainVideoElement.setAttribute('data-track', mainTrackIndex);
  mainVideoElement.onplay = () => adjustAnnotationsSize(playerElementId);
  mainPlayerElement.appendChild(mainVideoElement);

  // Needed so that we can match annotations canvas size to video client size.
  observeElementChanges(playerElementId);


  // add video thumbnail
  if (thumbnailTrackIndex !== undefined) {
    mainThumbnailElement = document.createElement('video');
    mainThumbnailElement.id = videoThumbnailId(playerElementId);
    mainThumbnailElement.className = "StreamVideoThumbnail";
    mainThumbnailElement.style.touchAction = 'none';
    mainThumbnailElement.setAttribute('data-track', thumbnailTrackIndex);
    mainPlayerElement.appendChild(mainThumbnailElement);
  }

  maybeConnectMainPlayer();
}

function setExtraVideoPlayer(playerElementId, trackIndex) {
  var playerElement = document.getElementById(playerElementId);
  var videoElement = document.createElement('video');
  videoElement.id = videoElementId(playerElementId);
  videoElement.className = "StreamVideo";
  videoElement.style.touchAction = 'none';
  videoElement.setAttribute('data-track', trackIndex);
  videoElement.onplay = () => adjustAnnotationsSize(playerElementId);
  playerElement.appendChild(videoElement);
  extraVideoElements.push({element: videoElement, connected: false});

  // Needed so that we can match annotations canvas size to video client size.
  observeElementChanges(playerElementId);
  
  maybeConnectExtraPlayers();
}

function observeElementChanges(playerElementId) {
  var playerElement = document.getElementById(playerElementId);
  var containerElement = playerElement.parentElement.parentElement;
  var observer = new MutationObserver(unused_mutations => adjustAnnotationsSize(playerElementId));
  observer.observe(containerElement, {
    attributes: true
  });
  elementObservers.push(observer);
}

function adjustAnnotationsSize(playerElementId) {
  var annotations = document.getElementById(videoAnnotationsId(playerElementId));
  var video = document.getElementById(videoElementId(playerElementId));
  if (video) {
    annotations.width = video.clientWidth;
    annotations.height = video.clientHeight;
  }
  drawAnnotations();
}

function drawAnnotations() {
  if (!window.camera_annotations) {
    return;
  }

  for (let id in window.camera_annotations) {
    let annotation = window.camera_annotations[id]
    let canvas = document.getElementById(id);
    let ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.strokeStyle = "red";
    annotation.boxes.forEach(box => {
      ctx.rect(
        box[1] * canvas.width,
        canvas.height - box[2] * canvas.height, // flip y.
        (box[3] - box[1]) * canvas.width,
        (box[2] - box[0]) * canvas.height);
    });
    ctx.stroke();
  }
}

// TODO: module-ify everything?
window.setMainVideoPlayer = setMainVideoPlayer;
window.setExtraVideoPlayer = setExtraVideoPlayer;
window.drawAnnotations = drawAnnotations;


function videoElementId(playerElementId) {
  return `${playerElementId}_video`;
}

function videoThumbnailId(playerElementId) {
  return `${playerElementId}_videoThumbnail`;
}

function videoAnnotationsId(playerElementId) {
  return `${playerElementId}_annotations`;
}

function maybeConnectMainPlayer() {
  if (!mainPlayerElement || !videoStreamer) {
    return;
  }
  registerMouseEvents(videoStreamer, mainVideoElement);
  videoStreamer.addVideoElement(mainVideoElement);

  if (mainThumbnailElement) {
    videoStreamer.addVideoElement(mainThumbnailElement);
  }
}

function maybeConnectExtraPlayers() {
  if (!videoStreamer) {
    return;
  }

  extraVideoElements.forEach(extraVideo => {
    if (!extraVideo.connected) {
      videoStreamer.addVideoElement(extraVideo.element);
    }
  });
}

async function setupVideoStreamer() {
  const videoStreamer = new VideoPlayer();
  await videoStreamer.setupConnection();
  videoStreamer.ondisconnect = () => onDisconnect();
  registerGamepadEvents(videoStreamer);
  // TODO: add these back, but make it so they only register when scene pane is selected
  //registerKeyboardEvents(videoStreamer);
  
  return videoStreamer;
}

function onDisconnect() {
  console.log("video streamer disconnected");
  videoStreamer = null;
  // TODO: clear state; schedule reconnect;
}

function initStreamer() {
  setupVideoStreamer().then(streamer => {
    videoStreamer = streamer;
    maybeConnectMainPlayer();
    maybeConnectExtraPlayers();
  })
}

initStreamer();