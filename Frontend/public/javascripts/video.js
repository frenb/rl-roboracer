import { VideoPlayer } from "./video-player.js";
import { registerGamepadEvents, registerKeyboardEvents, registerMouseEvents, sendClickEvent } from "./register-events.js";

var videoPlayers = {};

// TODO: module-ify everything?
function setupVideoStream (playerElementId) {
  showPlayButton(playerElementId);
  onClickPlayButton(playerElementId);
}

window.setupVideoStream = setupVideoStream;


window.document.oncontextmenu = function () {
  return false;     // cancel default menu
}

/*
Might have a use for this in the future*/
window.addEventListener('resize', function() {
  for (let player in videoPlayers) {
    if (videoPlayers[player]) {
      videoPlayers[player].resizeVideo();
    }
  }
}, true);


function playButtonId(playerElementId) {
  return `${playerElementId}_playButton`;
}

function videoElementId(playerElementId) {
  return `${playerElementId}_video`
}

function videoThumbnailId(playerElementId) {
  return `${playerElementId}_videoThumbnail`
}


function showPlayButton(playerElementId) {
  if (!document.getElementById(playButtonId(playerElementId))) {
    let elementPlayButton = document.createElement('img');
    elementPlayButton.id = playButtonId(playerElementId);
    elementPlayButton.src = 'images/Play.png';
    elementPlayButton.alt = 'Start Streaming';
    let playButton = document.getElementById(playerElementId).appendChild(elementPlayButton);
    playButton.addEventListener('click', () => onClickPlayButton(playerElementId));
  }
}

function onClickPlayButton(playerElementId) {

  var playButton = document.getElementById(playButtonId(playerElementId));

  if(playButton)
    playButton.style.display = 'none';

  const playerDiv = document.getElementById(playerElementId);

  // add video player
  const elementVideo = document.createElement('video');
  elementVideo.id = videoElementId(playerElementId);
  elementVideo.style.touchAction = 'none';
  playerDiv.appendChild(elementVideo);

  // add video thumbnail
  const elementVideoThumb = document.createElement('video');
  elementVideoThumb.id = videoThumbnailId(playerElementId);
  elementVideoThumb.style.touchAction = 'none';
  playerDiv.appendChild(elementVideoThumb);

  setupVideoPlayer([elementVideo, elementVideoThumb], undefined /* config */, playerElementId)
    .then(value => videoPlayers[playerElementId] = value);

  // TODO: full screen support?
  // add fullscreen button
  /*
  const elementFullscreenButton = document.createElement('img');
  elementFullscreenButton.id = 'fullscreenButton';
  elementFullscreenButton.src = 'images/FullScreen.png';
  playerDiv.appendChild(elementFullscreenButton);
  elementFullscreenButton.addEventListener ("click", function() {
    if (!document.fullscreenElement) {
      if(document.documentElement.requestFullscreen) {
        document.documentElement.requestFullscreen();
      }
      else if(document.documentElement.webkitRequestFullscreen){
        document.documentElement.webkitRequestFullscreen(Element.ALLOW_KEYBOARD_INPUT);
      }
    }
  });
  document.addEventListener('webkitfullscreenchange', onFullscreenChange);
  document.addEventListener('fullscreenchange', onFullscreenChange);

  function onFullscreenChange(e) {
    if(document.webkitFullscreenElement || document.fullscreenElement) {
      elementFullscreenButton.style.display = 'none';
    }
    else {
      elementFullscreenButton.style.display = 'block';
    }
  }
  */
}

async function setupVideoPlayer(elements, config, playerElementId) {
  const videoPlayer = new VideoPlayer(elements, config);
  await videoPlayer.setupConnection();

  videoPlayer.ondisconnect = () => onDisconnect(playerElementId);
  registerGamepadEvents(videoPlayer);
  registerKeyboardEvents(videoPlayer);
  registerMouseEvents(videoPlayer, elements[0]);
  
  return videoPlayer;
}

function onDisconnect(playerElementId) {
  const playerDiv = document.getElementById(playerElementId)
  clearChildren(playerDiv);
  delete videoPlayers[playerElementId];
  showPlayButton();
}

function clearChildren(element) {
  while (element.firstChild) {
    element.removeChild(element.firstChild);
  }
}
