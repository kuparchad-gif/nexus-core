/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import React, { useRef, useState, useEffect } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import Spinner from './Spinner';

interface ModelViewerProps {
    url: string;
}

const ModelViewer: React.FC<ModelViewerProps> = ({ url }) => {
    const mountRef = useRef<HTMLDivElement>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);
    const [progress, setProgress] = useState(0);

    useEffect(() => {
        if (!mountRef.current || !url) return;
        const currentMount = mountRef.current;

        setError(null);
        setLoading(true);
        setProgress(0);

        const scene = new THREE.Scene();
        
        const camera = new THREE.PerspectiveCamera(75, currentMount.clientWidth / currentMount.clientHeight, 0.1, 1000);
        camera.position.z = 5;

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setClearColor(0x000000, 0); // Transparent background
        renderer.setSize(currentMount.clientWidth, currentMount.clientHeight);
        currentMount.innerHTML = '';
        currentMount.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        const ambientLight = new THREE.AmbientLight(0xffffff, 2.5);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 3.5);
        directionalLight.position.set(5, 10, 7.5);
        scene.add(directionalLight);

        const loader = new GLTFLoader();
        loader.load(
            url,
            (gltf) => {
                const model = gltf.scene;
                // Center model
                const box = new THREE.Box3().setFromObject(model);
                const center = box.getCenter(new THREE.Vector3());
                model.position.sub(center);
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const fov = camera.fov * (Math.PI / 180);
                let cameraZ = Math.abs(maxDim / 2 * Math.tan(fov * 2));
                cameraZ *= 1.5; // zoom out a bit
                camera.position.z = cameraZ;
                camera.updateProjectionMatrix();

                scene.add(model);
                setLoading(false);
            },
            (xhr) => {
                // Progress callback
                if (xhr.total > 0) {
                    setProgress(Math.round((xhr.loaded / xhr.total) * 100));
                }
            }, 
            (err) => {
                console.error(`Error loading GLTF model from URL: ${url}`, err);
                setError('Failed to load 3D model. This could be due to an invalid URL, a network issue, or server Cross-Origin (CORS) restrictions. Please verify the asset source.');
                setLoading(false);
            }
        );

        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        const handleResize = () => {
            camera.aspect = currentMount.clientWidth / currentMount.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(currentMount.clientWidth, currentMount.clientHeight);
        };
        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            controls.dispose();
        };

    }, [url]);

    return (
        <div className="w-full h-full relative">
            <div ref={mountRef} className="w-full h-full"></div>
            {loading && (
                <div className="absolute inset-0 flex items-center justify-center">
                     <div className="text-center p-4">
                        <Spinner />
                        <p className="mt-4 text-slate-300 font-medium animate-pulse">
                            Loading 3D Model... {progress > 0 && `${progress}%`}
                        </p>
                    </div>
                </div>
            )}
            {error && (
                <div className="absolute inset-0 flex items-center justify-center bg-red-900/50 p-4">
                    <p className="text-red-300 font-medium text-center">{error}</p>
                </div>
            )}
        </div>
    );
};

export default ModelViewer;