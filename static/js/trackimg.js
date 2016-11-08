'use strict';

function fileUpload(form, action_url, div_id) {
    // Create the iframe...
    debugger;

		var iframe = document.createElement("iframe");
    iframe.setAttribute("id", "upload_iframe");
    iframe.setAttribute("name", "upload_iframe");
    iframe.setAttribute("width", "0");
    iframe.setAttribute("height", "0");
    iframe.setAttribute("border", "0");
    iframe.setAttribute("style", "width: 0; height: 0; border: none;");

    // Add to document...
    form.parentNode.appendChild(iframe);
    window.frames['upload_iframe'].name = "upload_iframe";

    var iframeId = document.getElementById("upload_iframe");

    // Add event...
    var eventHandler = function () {
						debugger;

            if (iframeId.detachEvent) iframeId.detachEvent("onload", eventHandler);
            else iframeId.removeEventListener("load", eventHandler, false);
						
						var content;

            // Message from server...
            if (iframeId.contentDocument) {
                content = iframeId.contentDocument.body.innerHTML;
            } else if (iframeId.contentWindow) {
                content = iframeId.contentWindow.document.body.innerHTML;
            } else if (iframeId.document) {
                content = iframeId.document.body.innerHTML;
            }

						// serverside contents
            // document.getElementById(div_id).innerHTML = content;

						document.getElementById(div_id).innerHTML="<img id='trackimg' width='128' height='128' />"

						// img spread
						var imgname = document.getElementById('tffile').files[0].name;
						document.getElementById('trackimg').src="{{ url_for('static',filename='uploads/') }}" + imgname;


            // Del the iframe...
            // setTimeout('iframeId.parentNode.removeChild(iframeId)', 250);
        }

    if (iframeId.addEventListener) iframeId.addEventListener("load", eventHandler, true);
    if (iframeId.attachEvent) iframeId.attachEvent("onload", eventHandler);

    // Set properties of form...
    form.setAttribute("target", "upload_iframe");
    form.setAttribute("action", action_url);
    form.setAttribute("method", "post");
    form.setAttribute("enctype", "multipart/form-data");
    form.setAttribute("encoding", "multipart/form-data");

    // Submit the form...
    form.submit();

		debugger;
    document.getElementById(div_id).innerHTML = "Uploading...";
}