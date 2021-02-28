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

async function runEditorScripts() {
    // TODO: assumes only one editor, no error handling if "start" function not defined.
    eval(editors[0].getValue());
    await start();
}