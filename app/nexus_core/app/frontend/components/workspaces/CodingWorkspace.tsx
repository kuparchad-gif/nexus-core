/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React from 'react';
import VirtualAppStream from '../VirtualAppStream';

const CodingWorkspace: React.FC = () => {
    return (
        <div className="w-full h-full p-4">
            <div className="w-full h-full glass-panel rounded-2xl overflow-hidden">
                <VirtualAppStream appName="VS Code" appUrl="https://vscode.dev" />
            </div>
        </div>
    );
};

export default CodingWorkspace;