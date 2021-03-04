var editors = {}

async function setWorkspace(workspace_name) {
    let workspaces = await jQuery.getJSON('workspaces/workspaces.json').promise();
    let workspace = workspaces[workspace_name];
    console.log("workspaces = " + JSON.stringify(workspace, null, 3));

    workspace.sources.forEach(file => {
        let editor = {};
        editor.get = file.get;
        editor.post = file.post;
        editors[file.name] = editor;
    });
}

function configureEditor(id) {
    var editor = editors[id];
    editor.ace_editor = ace.edit(id, {
        theme: "ace/theme/tomorrow_night_eighties",
        mode: "ace/mode/javascript",
        maxLines: 1000,
        wrap: true,
        autoScrollEditorIntoView: true
    });
    
    // Fetch contents.
    jQuery.get(editor.get, function(data) {
        editor.ace_editor.setValue(data, -1);
    });
}

function constructProgram() {
    let program_string = "";
    Object.keys(editors).forEach(id => {
        program_string += editors[id].ace_editor.getValue() + "\n"
    });
    return program_string;
}

async function runEditorScripts() {
    // Functions for script to console pane.
    // TODO: Override console.log.
    function log(msg) {
        let div = document.getElementById(window.program_log_div);
        let text = window.program_log_text || "";
        text += "[INFO] " + msg + "\n";
        window.program_log_text = text;
        div.innerHTML = '<pre>' + text + '</pre>';
        var parent_content = div.parentElement;
        parent_content.scrollTop = parent_content.scrollHeight;
    }

    function logError(e) {
        let div = document.getElementById(window.program_log_div);
        let text = window.program_log_text || "";
        text += '<span style=\"color:red\">' + "[ERROR] " + e + '\n</span>';
        window.program_log_text = text;
        div.innerHTML = '<pre>' + text + '</pre>';
        var parent_content = div.parentElement;
        parent_content.scrollTop = parent_content.scrollHeight;
    }

    // TODO: run this in an invisible iframe.
    eval(constructProgram());

    log('Program started...');
    try {
        await start();
    } catch (e) {
        logError(e);
    }
    log('Finished execution.');
}