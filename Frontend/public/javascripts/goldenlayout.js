var config = {
    content: [
        {
          type: 'row',
          content: [
            {
                type: 'column',
                content: [
                  {
                    type: 'stack',
                    content: [
                    ]
                  }
                ]
            },
            {
              type: 'column',
              content: [
                {
                  type: 'stack',
                  content: [
                    {
                      type: 'component',
                      componentName: 'streamPlayerComponent',
                      componentState: { id: "ScenePlayer", title: "scene", isMain: true, track: 0}
                    },
                    {
                      type: 'component',
                      componentName: 'streamPlayerComponent',
                      componentState: { id: "OverheadCameraPlayer", title: "camera/overhead", isMain: false, track: 1}
                    },
                  ]
                },
                {
                  type: 'stack',
                  activeItemIndex: 1,
                  content: [
                    {
                      type: 'component',
                      componentName: 'programLogComponent',
                      componentState: { id: "program_log_output" }
                    },
                    {
                      type: 'component',
                      componentName: 'rosLogComponent',
                      componentState: { id: "ros_log_output" }
                    }
                  ]
                }
              ]
            }
          ]
        }
    ]
};

var myLayout = new window.GoldenLayout(config, $('#golden_layout'));

var editorComponent = function(container, componentState) {
    console.log("editorComponent: " + componentState.id);
    container.setTitle(componentState.id);

    // Unsaved changes toggle.
    container.setSaved = function (saved) {
      if (saved) {
        this.setTitle(componentState.id);
      }
      else {
        this.setTitle('*' + componentState.id);
      }
    };

    container.getElement().html(`<div id="${componentState.id}"></div>`);

    container.on('resize', () => {
      console.log('resize ' + componentState.id);
      // TODO: hack? Need to call resize on editor when opened for the first time, but this needs to happen
      // slightly after on('show') when the element is actually visible.
      setTimeout(() => sources[componentState.id].ace_editor.resize(), 100);
    });

    container.on('destroy', () => {
      console.log('destroy ' + componentState.id);
      removeEditor(componentState.id);
    });

    container.on('show', () => {
      console.log('show ' + componentState.id);
      // Ensure we are actually in DOM. There are spurious shows when moving tabs.
      if ($('#' + componentState.id)) {
        addEditor(componentState.id, container);
        // TODO: hack? Need to call resize on editor when opened for the first time, but this needs to happen
        // slightly after on('show') when the element is actually visible.
        setTimeout(() => sources[componentState.id].ace_editor.resize(), 100);
      }
    });
}

var streamPlayerComponent = function(container, componentState) {
  console.log("streamPlayerComponent: " + componentState.id);
  container.setTitle(componentState.title);
  container.getElement().html(`
    <div id="${componentState.id}" class="StreamPlayer" style="z-index: 0;"></div>
    <canvas id="${componentState.id}_annotations" style="height: 100%; width: 100%; z-index: 1; position: absolute;"></canvas>`);
  if (componentState.isMain) {
    container.on('open', () => {
      window.setMainVideoPlayer(componentState.id, componentState.track);
    });
  } else {
    container.on('open', () => {
      window.setExtraVideoPlayer(componentState.id, componentState.track);
    });
  }
}

var rosLogComponent = function(container, componentState) {
  console.log("rosLogComponent: " + componentState.id);
  container.setTitle("ROS Logs");
  container.getElement().html(`<div style="color:white" id="${componentState.id}"></div>`);
  container.on('open', () => {
    window.ros_log_div = componentState.id;
  });
}

var programLogComponent = function(container, componentState) {
  console.log("programLogComponent: " + componentState.id);
  container.setTitle("Program");
  container.getElement().html(`<div style="color:white" id="${componentState.id}"></div>`);
  container.on('open', () => {
    window.program_log_div = componentState.id;
  });
}

var simpleComponent = function(container, componentState) {
    const newChild = document.createElement("h2");
    newChild.innerText = componentState.label;
    container
      .getElement()
      .get(0)
      .appendChild(newChild);
}

myLayout.registerComponent('editorComponent', editorComponent);
myLayout.registerComponent('programLogComponent', programLogComponent);
myLayout.registerComponent('streamPlayerComponent', streamPlayerComponent);
myLayout.registerComponent('rosLogComponent', rosLogComponent);
myLayout.registerComponent('simpleComponent', simpleComponent);

myLayout.init();

// TODO: allow loading a different workspace. And don't replace contents on refresh.
// Load default workspace.
myLayout.on('initialised', async function(event) {
  // Fetch workspace.
  //await setWorkspace("Pick & Place");
  //await setWorkspace("Pole & Cart TF");
  //await setWorkspace("Pole & Cart Python");
  await setWorkspace("Find & Pick");
  
  // Create editor windows.
  var editorsContainer = myLayout.root.contentItems[0].contentItems[0].contentItems[0];
  var sourceIds = Object.keys(sources);
  sourceIds.forEach(id => {
    console.log("adding editor component for " + id);
    let config =  {
      type: 'component',
      componentName: 'editorComponent',
      componentState: { id: id }
    };
    editorsContainer.addChild(config);
  });
});



