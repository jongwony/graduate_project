'use strict';

// ajax style
function fileUpload(trigger, info) {
				// create iframe
				var iframe = document.createElement("iframe");
				iframe.width=0;
				iframe.height=0;
				iframe.id='upload';

				// add to document
				var iframeId = trigger.parentNode.appendChild(iframe);
				
				// create form
				var form = document.createElement("form");
				
				// clone input
				var input = document.createElement("input");
				input = document.querySelector("input");
				
				// filename extract
				var filename = document.querySelector('input[type=file]').files[0].name;

				// add to document
				form.method="POST";
				form.action="/upload/"+info+"?filename="+filename;
				form.enctype="multipart/form-data";
				form.appendChild(input);

				var childDocumentBody = trigger.parentNode.children["upload"].contentWindow.document.body;
				childDocumentBody.appendChild(form);
				

				// auto submit(upload)
				form.submit();

				// remove iframe
				iframeId.parentNode.removeChild(iframeId);

				var imgpath = "/static/uploads/" + filename;
				var img = document.getElementById('tfimage');
				img.src = imgpath;
				img.width=128;
				img.height=128;

}

