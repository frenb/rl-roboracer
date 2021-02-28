var config = {
    content: [
        {
          type: 'row',
          content: [
            {
                type: 'column',
                content: [
                    {
                        type: 'component',
                        componentName: 'iframeComponent',
                        componentState: { src: 'http://localhost:3000/editor' }
                    }
                ]
            },
            {
              type: 'column',
              content: [
                {
                  type: 'component',
                  componentName: 'iframeComponent',
                  componentState: { src: 'http://localhost:80/videoplayer' }
                },
                {
                  type: 'component',
                  componentName: 'iframeComponent',
                  componentState: { src: "http://localhost:3000/logs" }
                }
              ]
            }
          ]
        }
    ]
};
var myLayout = new GoldenLayout( config );

var iframeComponent = function(container, componentState) {
    container.on('resize', () => {
      const iframe = container.getElement().get(0).childNodes[0];
      iframe.width = container.width;
      iframe.height = container.height;
    });
    // This code seems to run only once; attach .on event handlers to react to changes,
    // don't expect this code to be rerun.
    console.log("componentState.src: " + componentState.src);
    const newChild = document.createElement("iframe")
    newChild.frameBorder=0;
    newChild.style = "background:white;"
    newChild.src=componentState.src;
    container
      .getElement()
      .get(0)
      .appendChild(newChild);
}

var simpleComponent = function(container, componentState) {
    const newChild = document.createElement("h2");
    newChild.innerText = componentState.label;
    container
      .getElement()
      .get(0)
      .appendChild(newChild);
}

myLayout.registerComponent( 'iframeComponent', iframeComponent);
myLayout.registerComponent('simpleComponent', simpleComponent);

myLayout.init();

