/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const NexusIdleAnimation: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!mountRef.current) return;

    const currentMount = mountRef.current;
    let controls: any;

    // Scene setup
    const scene = new THREE.Scene();
    
    const camera = new THREE.PerspectiveCamera(75, currentMount.clientWidth / currentMount.clientHeight, 0.1, 1000);
    camera.position.z = 25;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setClearColor(0x000000, 0); // Transparent background
    renderer.setSize(currentMount.clientWidth, currentMount.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    currentMount.appendChild(renderer.domElement);
    
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enableZoom = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.4;

    // Metatron's Cube Structure
    const group = new THREE.Group();
    const material = new THREE.LineBasicMaterial({ color: 0x4f46e5 }); // indigo-600
    const nodeGeometry = new THREE.SphereGeometry(0.3, 16, 16);
    const nodeMaterial = new THREE.MeshBasicMaterial({ color: 0x312e81 }); // indigo-800

    // Define 13 node positions based on Metatron's Cube geometry
    const R = 10; // Outer radius
    const r = 5;  // Inner radius
    const nodes = [
      new THREE.Vector3(0, 0, 0), // 0: Center
      // Inner hexagon (Fruit of Life)
      new THREE.Vector3(r, 0, 0), // 1
      new THREE.Vector3(r * Math.cos(Math.PI / 3), r * Math.sin(Math.PI / 3), 0), // 2
      new THREE.Vector3(r * Math.cos(2 * Math.PI / 3), r * Math.sin(2 * Math.PI / 3), 0), // 3
      new THREE.Vector3(-r, 0, 0), // 4
      new THREE.Vector3(r * Math.cos(4 * Math.PI / 3), r * Math.sin(4 * Math.PI / 3), 0), // 5
      new THREE.Vector3(r * Math.cos(5 * Math.PI / 3), r * Math.sin(5 * Math.PI / 3), 0), // 6
      // Outer hexagon
      new THREE.Vector3(R, 0, 0), // 7
      new THREE.Vector3(R * Math.cos(Math.PI / 3), R * Math.sin(Math.PI / 3), 0), // 8
      new THREE.Vector3(R * Math.cos(2 * Math.PI / 3), R * Math.sin(2 * Math.PI / 3), 0), // 9
      new THREE.Vector3(-R, 0, 0), // 10
      new THREE.Vector3(R * Math.cos(4 * Math.PI / 3), R * Math.sin(4 * Math.PI / 3), 0), // 11
      new THREE.Vector3(R * Math.cos(5 * Math.PI / 3), R * Math.sin(5 * Math.PI / 3), 0), // 12
    ];

    // Create spheres for nodes
    nodes.forEach(pos => {
      const nodeMesh = new THREE.Mesh(nodeGeometry, nodeMaterial);
      nodeMesh.position.copy(pos);
      group.add(nodeMesh);
    });

    // Create edges
    const edges = [
      // Center to inner
      [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6],
      // Inner hexagon
      [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1],
      // Center to outer
      [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12],
      // Outer hexagon
      [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 7],
      // Star of David / Inter-hex connections
      [1, 3], [3, 5], [5, 1],
      [2, 4], [4, 6], [6, 2],
      // Connections for Platonic Solids
      [7, 9], [9, 11], [11, 7],
      [8, 10], [10, 12], [12, 8],
      [1, 8], [2, 9], [3, 10], [4, 11], [5, 12], [6, 7]
    ];

    edges.forEach(edge => {
      const geometry = new THREE.BufferGeometry().setFromPoints([nodes[edge[0]], nodes[edge[1]]]);
      const line = new THREE.Line(geometry, material);
      group.add(line);
    });
    
    scene.add(group);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
      if (currentMount) {
        camera.aspect = currentMount.clientWidth / currentMount.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(currentMount.clientWidth, currentMount.clientHeight);
      }
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (currentMount && renderer.domElement) {
        currentMount.removeChild(renderer.domElement);
      }
      controls.dispose();
    };
  }, []);

  return <div ref={mountRef} className="w-full h-full" />;
};

export default NexusIdleAnimation;