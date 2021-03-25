import { VideoPlayer } from "./video-player.js";
import { registerGamepadEvents, registerKeyboardEvents, registerMouseEvents, sendClickEvent } from "./register-events.js";

var videoStreamer;
var mainPlayerElement;
var mainVideoElement;
var mainThumbnailElement;
var extraPlayerElements = [];


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
  mainThumbnailElement = document.createElement('video');
  mainThumbnailElement.id = videoThumbnailId(playerElementId);
  mainThumbnailElement.className = "StreamVideoThumbnail";
  mainThumbnailElement.style.touchAction = 'none';
  mainThumbnailElement.setAttribute('data-track', thumbnailTrackIndex);
  mainPlayerElement.appendChild(mainThumbnailElement);

  maybeConnectMainPlayer();
}

// TODO: module-ify everything?
window.setMainVideoPlayer = setMainVideoPlayer;


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
  videoStreamer.addVideoElement(mainThumbnailElement);
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
  // TODO: schedule reconnect;
}

function initStreamer() {
  setupVideoStreamer().then(streamer => {
    videoStreamer = streamer;
    maybeConnectMainPlayer();
  })
}

initStreamer();