/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useEffect, useRef } from 'react';

interface P5ViewerProps {
  code: string;
}

const P5Viewer: React.FC<P5ViewerProps> = ({ code }) => {
  const iframeRef = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    if (!iframeRef.current) return;

    // By simply injecting the user's code into a script tag after p5.js is loaded,
    // we leverage p5.js's "global mode". It will automatically find and execute
    // standard p5 functions like setup() and draw(). The try...catch block ensures
    // that any syntax errors in the generated code are caught and displayed in the
    // iframe, rather than crashing the main React application.
    const htmlContent = `
      <!DOCTYPE html>
      <html>
        <head>
          <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.4/p5.min.js"></script>
          <style>
            body { margin: 0; overflow: hidden; background-color: #ffffff; }
            canvas { display: block; }
          </style>
        </head>
        <body>
          <script>
            try {
              ${code}
            } catch (err) {
              document.body.innerHTML = '<div style="font-family: monospace; color: #c0392b; padding: 1rem; white-space: pre-wrap;">' + err.message + '</div>';
              console.error(err);
            }
          </script>
        </body>
      </html>
    `;

    const iframe = iframeRef.current;
    iframe.srcdoc = htmlContent;

  }, [code]);

  return (
    <div className="w-full h-full bg-white">
      <iframe
        ref={iframeRef}
        title="p5.js Sketch"
        style={{ width: '100%', height: '100%', border: 'none' }}
        sandbox="allow-scripts"
      />
    </div>
  );
};

export default React.memo(P5Viewer);