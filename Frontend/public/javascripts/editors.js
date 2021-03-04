var editors = []

function configureEditor(id) {
    var editor = ace.edit(id, {
        theme: "ace/theme/tomorrow_night_eighties",
        mode: "ace/mode/javascript",
        maxLines: 1000,
        wrap: true,
        autoScrollEditorIntoView: true
    });
    
    // Set default value.
    jQuery.get('javascripts/pick_and_place.js', function(data) {
        editor.setValue(data, -1);
    });

    editors.push(editor);
}

function constructProgram() {
    let program_string = "";
    editors.forEach(editor => program_string += editor.getValue() + "\n");
    return program_string;
}

async function runEditorScripts() {
    // Functions for script to console pane.
    // TODO: Override console.log.
    function log(msg) {
        let div = document.getElementById(window.program_log_div);
        let content = div.innerHTML;
        content += '<pre>' + "[INFO] " + msg + '\n</pre>'
        div.innerHTML = content;
        var parent_content = div.parentElement;
        parent_content.scrollTop = parent_content.scrollHeight;
    }

    function logError(e) {
        let div = document.getElementById(window.program_log_div);
        let content = div.innerHTML;
        content += '<pre><span style=\"color:red\">' + "[ERROR] " + e + '\n</span></pre>'
        div.innerHTML = content;
        var parent_content = div.parentElement;
        parent_content.scrollTop = parent_content.scrollHeight;
    }

    // TODO: run this in an invisible iframe.
    eval(constructProgram());

    if (start == null) {
        logError('Program does not define an entry point.');
        return;
    }

    log('Program started...');
    await start();
    log('Finished execution.');
}