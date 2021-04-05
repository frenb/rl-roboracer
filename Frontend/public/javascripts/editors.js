var sources = {};
var active_workspace = null;

async function setWorkspace(workspace_name) {
    let workspaces = await jQuery.getJSON('workspaces/workspaces.json').promise();
    let workspace = workspaces[workspace_name];
    active_workspace = workspace;
    console.log("workspaces = " + JSON.stringify(workspace, null, 3));

    if (workspace.type == "python") {
        await connectPythonWorkspace()
    }

    for (var remoteSource of workspace.sources) {
        let source = {};
        source.get = remoteSource.get;
        source.post = remoteSource.post;
        source.txt = await jQuery.get(remoteSource.get).promise();
        console.log("downloaded source for " + remoteSource.name);
        sources[remoteSource.name] = source;
    }
}

// TODO: dont connect if already connected
async function connectPythonWorkspace() {
    socket.emit('python_workspace');
    socket.on('python_output', function(data) {
        if (window.program_log_div) {
            let div = document.getElementById(window.program_log_div);
            let text = window.program_log_text || "";
            text += data.data;
            window.program_log_text = text;
            div.innerHTML = '<pre>' + ansi_up.ansi_to_html(text) + '</pre>';
            var parent_content = div.parentElement;
            parent_content.scrollTop = parent_content.scrollHeight;
        }
    });
    socket.on('python_annotation', function(data) {
        console.log(`received camera annotation ${data.data}`);
        let annotation = JSON.parse(data.data);
        let cameraDivId = annotation.annotateCamera;
        window.camera_annotations = window.camera_annotations || {};
        window.camera_annotations[cameraDivId] = annotation;
        window.drawAnnotations();
    });
}

function saveEditors() {
    Object.keys(sources).forEach(async function(id) {
        if (sources[id].ace_editor) {
            sources[id].txt = sources[id].ace_editor.getValue();
        }

        await $.ajax({
            type: "POST",
            url: sources[id].post,
            data: JSON.stringify({ txt: sources[id].txt }),
            contentType: 'application/json'
        }).promise();

        if (sources[id].editorContainer) {
            sources[id].editorContainer.setSaved(true);
        }
    })
}

function removeEditor(id) {
    sources[id].ace_editor.destroy();
    sources[id].ace_editor = null;
    sources[id].editorContainer = null;
}

function addEditor(id, editorContainer) {
    var source = sources[id];

    if (source.ace_editor) {
        console.log("editor already exists");
        return;
    }

    source.ace_editor = ace.edit(id, {
        theme: "ace/theme/tomorrow_night_eighties",
        mode: `ace/mode/${active_workspace.type}`,
        maxLines: 1000,
        wrap: true,
        autoScrollEditorIntoView: true
    });
    source.ace_editor.getSession().setUndoManager(new ace.UndoManager())
    source.ace_editor.setValue(sources[id].txt, -1);
    source.editorContainer = editorContainer;
    console.log("editor created " + id);

    source.ace_editor.on('change', (_) => editorContainer.setSaved(false));
}

async function switchWorkspace(workspace_name)
/*
 //await setWorkspace("Pick & Place");
  //await setWorkspace("Pole & Cart TF");
  await setWorkspace("Pole & Cart Python");
  //await setWorkspace("Find & Pick");
*/
{
   await setWorkspace(workspace_name);
   console.log("switch completed");
}

function constructProgram() {
    let program_string = "";
    Object.keys(sources).forEach(id => {
        program_string += sources[id].txt + "\n"
    });
    return program_string;
}

async function runProgram() {
    if (active_workspace.type == "javascript") {
        await runJsProgram();
    } else if (active_workspace.type == "python") {
        await runPyProgram();
    }
}

async function runPyProgram() {
    socket.emit("python_run", active_workspace.main)
}

async function stopProgram() {
    if (active_workspace.type == "python") {
        await stopPyProgram();
    }
}

async function stopPyProgram() {
    socket.emit("python_stop");
}

async function runJsProgram() {
    // Functions for script to console pane.
    // TODO: Override console.log.
    function log(msg) {
        let div = document.getElementById(window.program_log_div);
        let text = window.program_log_text || "";
        text += "[INFO] " + msg + "\n";
        window.program_log_text = text;
        if (div) {
            div.innerHTML = '<pre>' + text + '</pre>';
            var parent_content = div.parentElement;
        }
        parent_content.scrollTop = parent_content.scrollHeight;
    }

    function logError(e) {
        let div = document.getElementById(window.program_log_div);
        let text = window.program_log_text || "";
        text += '<span style=\"color:red\">' + "[ERROR] " + e + '\n</span>';
        window.program_log_text = text;
        if (div) {
            div.innerHTML = '<pre>' + text + '</pre>';
            var parent_content = div.parentElement;
        }
        parent_content.scrollTop = parent_content.scrollHeight;
    }

    // TODO: run this in an invisible iframe.
    eval(constructProgram());

    log('Program started...');
    try {
        await start();
    } catch (e) {
        logError(e);
        // G_CHECK remove
        throw(e);
    }
    log('Finished execution.');
}