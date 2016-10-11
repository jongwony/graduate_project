'use strict';

function fileUpload(button) {
				// create iframe
				debugger;
				var iframe = document.createElement("iframe");
				iframe.width=240;
				console.log(iframe.width);
				iframe.height=240;
				iframe.id='upload';
				iframe.src='/test';
			
				button.parentNode.appendChild(iframe);
				
				setTimeout('button.parentNode.removeChild(iframe)',1000);

				var img = document.createElement('img');
				img.id = 'img';
				img.width = 128;
				img.height = 128;
				img.src="{{ url_for('static', filename='uploads/_Detection_') }}";

				button.parentNode.appendChild(img);



				var iframeId = document.getElementById('upload');
				console.log(iframeId);
}


