import { useRef, useMemo, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

interface VoxelData {
  gridSize: number;
  data: Uint8Array;
}

interface VoxelMeshProps {
  voxelData: VoxelData;
  threshold?: number;
  color?: string;
  opacity?: number;
  rotate?: boolean;
  sigmoidOpacity?: boolean;
}

// Sigmoid function for opacity mapping
function sigmoid(x: number, steepness: number = 10, midpoint: number = 0.5): number {
  return 1 / (1 + Math.exp(-steepness * (x - midpoint)));
}

function VoxelMesh({
  voxelData,
  threshold = 0.1,
  color = "#ebcb8b",
  opacity = 0.8,
  rotate = true,
  sigmoidOpacity = true,
}: VoxelMeshProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const groupRef = useRef<THREE.Group>(null);

  // Extract voxels above threshold
  const { positions, colors, opacities, count } = useMemo(() => {
    const { gridSize, data } = voxelData;
    const thresholdValue = threshold * 255;

    const positionsList: number[][] = [];
    const colorsList: number[][] = [];
    const opacitiesList: number[] = [];

    // Color gradient from low to high values
    const lowColor = new THREE.Color(color).multiplyScalar(0.5);
    const highColor = new THREE.Color(color);

    for (let z = 0; z < gridSize; z++) {
      for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
          const idx = x + y * gridSize + z * gridSize * gridSize;
          const value = data[idx];

          if (value > thresholdValue) {
            // Center the grid around origin
            const px = x / gridSize - 0.5;
            const py = y / gridSize - 0.5;
            const pz = z / gridSize - 0.5;
            positionsList.push([px, py, pz]);

            // Normalized value 0-1
            const t = value / 255;

            // Interpolate color based on value
            const c = new THREE.Color().lerpColors(lowColor, highColor, t);
            colorsList.push([c.r, c.g, c.b]);

            // Sigmoid opacity mapping
            const voxelOpacity = sigmoidOpacity
              ? sigmoid(t, 8, 0.3) * opacity
              : t * opacity;
            opacitiesList.push(voxelOpacity);
          }
        }
      }
    }

    return {
      positions: positionsList,
      colors: colorsList,
      opacities: opacitiesList,
      count: positionsList.length,
    };
  }, [voxelData, threshold, color, opacity, sigmoidOpacity]);

  // Set up instanced mesh
  useEffect(() => {
    if (!meshRef.current || count === 0) return;

    const mesh = meshRef.current;
    const matrix = new THREE.Matrix4();
    const colorAttr = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      const [x, y, z] = positions[i];
      matrix.setPosition(x, y, z);
      mesh.setMatrixAt(i, matrix);

      // Modulate color by opacity for visual effect
      const [r, g, b] = colors[i];
      const o = opacities[i];
      colorAttr[i * 3] = r * o + (1 - o) * 0.1;
      colorAttr[i * 3 + 1] = g * o + (1 - o) * 0.1;
      colorAttr[i * 3 + 2] = b * o + (1 - o) * 0.1;
    }

    mesh.geometry.setAttribute(
      "color",
      new THREE.InstancedBufferAttribute(colorAttr, 3)
    );
    mesh.instanceMatrix.needsUpdate = true;
    mesh.geometry.attributes.color.needsUpdate = true;
  }, [positions, colors, opacities, count]);

  // Rotate the group
  useFrame((_, delta) => {
    if (rotate && groupRef.current) {
      groupRef.current.rotation.y += delta * 0.3;
    }
  });

  if (count === 0) {
    return null;
  }

  const voxelSize = 0.8 / voxelData.gridSize;

  return (
    <group ref={groupRef}>
      <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
        <boxGeometry args={[voxelSize, voxelSize, voxelSize]} />
        <meshStandardMaterial
          vertexColors
          transparent
          opacity={opacity}
        />
      </instancedMesh>
      {/* Bounding box */}
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(1, 1, 1)]} />
        <lineBasicMaterial color="#d8dee9" />
      </lineSegments>
    </group>
  );
}

interface VoxelViewerProps {
  voxelData: VoxelData | null;
  title?: string;
  threshold?: number;
  color?: string;
  rotate?: boolean;
  sigmoidOpacity?: boolean;
}

export function VoxelViewer({
  voxelData,
  title,
  threshold = 0.1,
  color = "#ebcb8b",
  rotate = true,
  sigmoidOpacity = true,
}: VoxelViewerProps) {
  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      {title && (
        <div
          style={{
            position: "absolute",
            top: "8px",
            left: "50%",
            transform: "translateX(-50%)",
            color: "#eceff4",
            fontSize: "14px",
            fontWeight: 500,
            zIndex: 10,
            textShadow: "0 1px 2px rgba(0,0,0,0.5)",
          }}
        >
          {title}
        </div>
      )}
      <Canvas
        camera={{ position: [1.5, 1, 1.5], fov: 45 }}
        style={{ background: "#4c566a" }}
      >
        <ambientLight intensity={0.4} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />
        <directionalLight position={[-5, -5, -5]} intensity={0.3} />

        {voxelData && (
          <VoxelMesh
            voxelData={voxelData}
            threshold={threshold}
            color={color}
            rotate={rotate}
            sigmoidOpacity={sigmoidOpacity}
          />
        )}

        <OrbitControls
          enablePan={false}
          minDistance={1}
          maxDistance={5}
        />
      </Canvas>
    </div>
  );
}
