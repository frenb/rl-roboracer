import { VideoPlayer } from "./video-player.js";
import { registerGamepadEvents, registerKeyboardEvents, registerMouseEvents, sendClickEvent } from "./register-events.js";

var videoStreamer;
var mainPlayerElement;
var mainVideoElement;
var mainThumbnailElement;
var extraVideoElements = [];


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
  mainPlayerElement.appendChild(mainVideoElement);


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
  playerElement.appendChild(videoElement);
  extraVideoElements.push({element: videoElement, connected: false});
  
  maybeConnectExtraPlayers();
}

// TODO: module-ify everything?
window.setMainVideoPlayer = setMainVideoPlayer;
window.setExtraVideoPlayer = setExtraVideoPlayer;


function videoElementId(playerElementId) {
  return `${playerElementId}_video`
}

function videoThumbnailId(playerElementId) {
  return `${playerElementId}_videoThumbnail`
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
  registerKeyboardEvents(videoStreamer);
  
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