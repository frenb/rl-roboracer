import Signaling, { WebSocketSignaling } from "./signaling.js"

// enum type of event sending from Unity
var UnityEventType = {
  SWITCH_VIDEO: 0
};

export class VideoPlayer {
  constructor(config) {
    const _this = this;
    this.cfg = VideoPlayer.getConfiguration(config);
    this.pc = null;
    this.channel = null;
    this.offerOptions = {
      offerToReceiveAudio: true,
      offerToReceiveVideo: true,
    };
    this.connectionId = null;

    this.streams = [];
    this.videoElements = [];
    this.audio_track = null;

    this.videoTrackList = [];
    this.videoTrackIndex = 0;
    
    this.ondisconnect = function () { };
  }

  addVideoElement(element) {
    this.streams.push(new MediaStream());
    this.videoElements.push(element);
    element.playsInline = true;
    element.autoplay = true;
    element.muted = true;
    var _this = this;
    // Treat first as camera for scene. For main scene video only: we need a resize callback to properly translate
    // interaction positions.
    if (this.streams.length == 1) {
      if (this.audio_track) {
        this.streams[0].addTrack(this.audio_track);
      }
      this.maybeConnectVideoTrack(element.dataset.track);
      element.addEventListener('loadedmetadata', function () {
        element.play();
        _this.resizeVideo();
      });
    } else {
      element.addEventListener('loadedmetadata', function () {
        element.play();
      });
    }
  }

  static getConfiguration(config) {
    if (config === undefined) {
      config = {};
    }
    config.sdpSemantics = 'unified-plan';
    config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
    return config;
  }

  async setupConnection() {
    const _this = this;
    // close current RTCPeerConnection
    if (this.pc) {
      console.log('Close current PeerConnection');
      this.pc.close();
      this.pc = null;
    }

    // Decide Signaling Protocol
    
    const protocolEndPoint = location.protocol + '//' + "localhost:80" + '/' + 'protocol';
    console.log("**** "+protocolEndPoint);
    const createResponse = await fetch(protocolEndPoint);
    const res = await createResponse.json();

    if (res.useWebSocket) {
      this.signaling = new WebSocketSignaling();
    } else {
      this.signaling = new Signaling();
    }

    // Create peerConnection with proxy server and set up handlers
    this.pc = new RTCPeerConnection(this.cfg);
    this.pc.onsignalingstatechange = function (e) {
      console.log('signalingState changed:', e);
    };
    this.pc.oniceconnectionstatechange = function (e) {
      console.log('iceConnectionState changed:', e);
      console.log('pc.iceConnectionState:' + _this.pc.iceConnectionState);
      if (_this.pc.iceConnectionState === 'disconnected') {
        // TODO: Clear tracks from MediaStreamers
        _this.ondisconnect();
      }
    };
    this.pc.onicegatheringstatechange = function (e) {
      console.log('iceGatheringState changed:', e);
    };
    this.pc.ontrack = function (e) {
      if(e.track.kind == 'video') {
        _this.videoTrackList.push(e.track);
        console.log(`G_CHECK video tracks: ${_this.videoTrackList.length}`);
        _this.maybeConnectVideoTrack(_this.videoTrackList.length - 1 /* new track's index */);
      }
      if(e.track.kind == 'audio') {
        _this.audio_track = e.track;
        _this.maybeConnectAudioTrack();
      }
    };
    this.pc.onicecandidate = function (e) {
      if (e.candidate != null) {
        _this.signaling.sendCandidate(_this.connectionId, e.candidate.candidate, e.candidate.sdpMid, e.candidate.sdpMLineIndex);
      }
    };
    // Create data channel with proxy server and set up handlers
    this.channel = this.pc.createDataChannel('data');
    this.channel.onopen = function () {
      console.log('Datachannel connected.');
    };
    this.channel.onerror = function (e) {
      console.log("The error " + e.error.message + " occurred\n while handling data with proxy server.");
    };
    this.channel.onclose = function () {
      console.log('Datachannel disconnected.');
    };
    this.channel.onmessage = async (msg) => {
      // receive message from unity and operate message
      let data;
      // receive message data type is blob only on Firefox
      if(navigator.userAgent.indexOf('Firefox') != -1) {
        data = await msg.data.arrayBuffer();
      }else{
        data = msg.data;
      }
      const bytes = new Uint8Array(data);
      _this.videoTrackIndex = bytes[1];
      switch(bytes[0]) {
        case UnityEventType.SWITCH_VIDEO:
          // TODO: suport?
          console.error("UnityEventType.SWITCH_VIDEO not supported");
          // _this.switchVideo(_this.videoTrackIndex);
          break;
      }
    };

    this.signaling.addEventListener('answer', async (e) => {
      const answer = e.detail;
      const desc = new RTCSessionDescription({ sdp: answer.sdp, type: "answer" });
      await _this.pc.setRemoteDescription(desc);
    });

    this.signaling.addEventListener('candidate', async (e) => {
      const candidate = e.detail;
      const iceCandidate = new RTCIceCandidate({ candidate: candidate.candidate, sdpMid: candidate.sdpMid, sdpMLineIndex: candidate.sdpMLineIndex });
      await _this.pc.addIceCandidate(iceCandidate);
    });

    // setup signaling
    this.connectionId = await this.signaling.start();

    // Add transceivers to receive multi stream.
    // It can receive two video tracks and one audio track from Unity app.
    // This operation is required to generate offer SDP correctly.
    this.pc.addTransceiver('video', { direction: 'recvonly' });
    this.pc.addTransceiver('video', { direction: 'recvonly' });
    this.pc.addTransceiver('audio', { direction: 'recvonly' });

    // create offer
    const offer = await this.pc.createOffer(this.offerOptions);

    // set local sdp
    const desc = new RTCSessionDescription({ sdp: offer.sdp, type: "offer" });
    await this.pc.setLocalDescription(desc);
    await this.signaling.sendOffer(this.connectionId, offer.sdp);
  };

  resizeVideo() {
    console.log("inside resize video");
    const clientRect = this.videoElements[0].getBoundingClientRect();
    const videoRatio = this.videoWidth / this.videoHeight;
    const clientRatio = clientRect.width / clientRect.height;

    this._videoScale = videoRatio > clientRatio ? clientRect.width / this.videoWidth : clientRect.height / this.videoHeight;
    const videoOffsetX = videoRatio > clientRatio ? 0 : (clientRect.width - this.videoWidth * this._videoScale) * 0.5;
    const videoOffsetY = videoRatio > clientRatio ? (clientRect.height - this.videoHeight * this._videoScale) * 0.5 : 0;
    this._videoOriginX = clientRect.left + videoOffsetX;
    this._videoOriginY = clientRect.top + videoOffsetY;
  }

  // switch streaming destination main video and secondly video
  /*
  switchVideo(indexVideoTrack) {

    this.videoElements.forEach((element, i) => {
      element.srcObject = this.streams[i];
      this.replaceTrack(this.streams[i], this.videoTrackList[0].element.dataset.track);
    });
  }
   // replace video track related the MediaStream
  replaceTrack(stream, newTrack) {
    const tracks = stream.getVideoTracks();
    for(const track of tracks) {
      if(track.kind == 'video') {
        stream.removeTrack(track);
      }
    }
    stream.addTrack(newTrack);
  }
  
  */

  maybeConnectAudioTrack() {
    if (this.audio_track != null && this.streams.length > 1) {
      this.streams[0].addTrack(this.audio_track);
    }
  }

  maybeConnectVideoTrack(trackIndex) {
    if (this.videoTrackList.length <= trackIndex) {
      // Track not presented yet by server.
      return;
    }

    // Find matching video element.
    for (var i = 0; i < this.videoElements.length; i++) {
      let element = this.videoElements[i];
      if (element.dataset.track == trackIndex && this.streams[i].getVideoTracks().length == 0) {
        element.srcObject = this.streams[i];
        this.streams[i].addTrack(this.videoTrackList[trackIndex]);
      }
    }
  }

  get videoWidth() {
    return this.videoElements[0].videoWidth;
  }

  get videoHeight() {
    return this.videoElements[0].videoHeight;
  }

  get videoOriginX() {
    return this._videoOriginX;
  }

  get videoOriginY() {
    return this._videoOriginY;
  }

  get videoScale() {
    return this._videoScale;
  }

  close() {
    if (this.pc) {
      console.log('Close current PeerConnection');
      this.pc.close();
      this.pc = null;
    }
  };

  sendMsg(msg) {
    if (this.channel == null) {
      return;
    }
    switch (this.channel.readyState) {
      case 'connecting':
        console.log('Connection not ready');
        break;
      case 'open':
        this.channel.send(msg);
        break;
      case 'closing':
        console.log('Attempt to sendMsg message while closing');
        break;
      case 'closed':
        console.log('Attempt to sendMsg message while connection closed.');
        break;
    }
  };
}
