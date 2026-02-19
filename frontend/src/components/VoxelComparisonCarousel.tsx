import { useState, useEffect } from "react";
import pako from "pako";
import { VoxelViewer } from "./VoxelViewer";
import styles from "../styles/VoxelComparisonCarousel.module.css";

interface SampleMetadata {
  index: number;
  simHash: string;
  gridSize: number;
  valueRange: { min: number; max: number };
  file: string;
}

interface Metadata {
  batchName: string;
  numSamples: number;
  samples: SampleMetadata[];
}

interface VoxelData {
  gridSize: number;
  data: Uint8Array;
}

interface SampleData {
  groundTruth: VoxelData;
  prediction: VoxelData;
  metadata: SampleMetadata;
}

function ArrowIcon({ direction }: { direction: "left" | "right" }) {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {direction === "left" ? (
        <polyline points="15 18 9 12 15 6" />
      ) : (
        <polyline points="9 18 15 12 9 6" />
      )}
    </svg>
  );
}

async function loadSampleData(
  sampleMeta: SampleMetadata
): Promise<SampleData | null> {
  try {
    const response = await fetch(`/voxels/${sampleMeta.file}`);
    const rawData = await response.arrayBuffer();
    const uint8Data = new Uint8Array(rawData);

    // Check if data is gzipped (magic bytes 0x1f 0x8b)
    let jsonString: string;
    if (uint8Data[0] === 0x1f && uint8Data[1] === 0x8b) {
      // Decompress gzipped data
      jsonString = pako.inflate(uint8Data, { to: "string" });
    } else {
      // Already decompressed (browser or server did it)
      jsonString = new TextDecoder().decode(uint8Data);
    }

    const jsonData = JSON.parse(jsonString);

    return {
      groundTruth: {
        gridSize: jsonData.gridSize,
        data: new Uint8Array(jsonData.groundTruth),
      },
      prediction: {
        gridSize: jsonData.gridSize,
        data: new Uint8Array(jsonData.prediction),
      },
      metadata: sampleMeta,
    };
  } catch (error) {
    console.error("Failed to load sample data:", error);
    return null;
  }
}

export function VoxelComparisonCarousel() {
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [sampleData, setSampleData] = useState<SampleData | null>(null);
  const [loading, setLoading] = useState(true);

  // Load metadata on mount
  useEffect(() => {
    async function loadMetadata() {
      try {
        const response = await fetch("/voxels/metadata.json");
        const data = await response.json();
        setMetadata(data);
      } catch (error) {
        console.error("Failed to load voxel metadata:", error);
      }
    }
    loadMetadata();
  }, []);

  // Load sample data when index changes
  useEffect(() => {
    if (!metadata || metadata.samples.length === 0) return;

    setLoading(true);
    loadSampleData(metadata.samples[currentIndex]).then((data) => {
      setSampleData(data);
      setLoading(false);
    });
  }, [metadata, currentIndex]);

  function goToPrevious() {
    if (!metadata) return;
    setCurrentIndex((prev) =>
      prev === 0 ? metadata.samples.length - 1 : prev - 1
    );
  }

  function goToNext() {
    if (!metadata) return;
    setCurrentIndex((prev) =>
      prev === metadata.samples.length - 1 ? 0 : prev + 1
    );
  }

  if (!metadata) {
    return (
      <div className={styles.carousel}>
        <div className={styles.loading}>Loading...</div>
      </div>
    );
  }

  return (
    <div className={styles.carousel}>
      <div className={styles.viewerContainer}>
        <button
          className={`${styles.navButton} ${styles.navButtonLeft}`}
          onClick={goToPrevious}
          aria-label="Previous sample"
        >
          <ArrowIcon direction="left" />
        </button>

        <div className={styles.viewers}>
          <div className={styles.viewer}>
            {loading ? (
              <div className={styles.loading}>Loading...</div>
            ) : (
              <VoxelViewer
                voxelData={sampleData?.groundTruth ?? null}
                title="Ground Truth"
                threshold={0.1}
                color="#a3be8c"
              />
            )}
          </div>
          <div className={styles.viewer}>
            {loading ? (
              <div className={styles.loading}>Loading...</div>
            ) : (
              <VoxelViewer
                voxelData={sampleData?.prediction ?? null}
                title="Prediction"
                threshold={0.1}
                color="#b48ead"
              />
            )}
          </div>
        </div>

        <button
          className={`${styles.navButton} ${styles.navButtonRight}`}
          onClick={goToNext}
          aria-label="Next sample"
        >
          <ArrowIcon direction="right" />
        </button>
      </div>

      <div className={styles.info}>
        <p className={styles.description}>
          Comparison of ground truth inclusion geometry (green) with neural
          network prediction (blue). The model learns to reconstruct 3D
          material distributions from boundary sensor measurements.
        </p>
      </div>

      <div className={styles.indicators}>
        {metadata.samples.map((_, index) => (
          <button
            key={index}
            className={`${styles.indicator} ${
              index === currentIndex ? styles.indicatorActive : ""
            }`}
            onClick={() => setCurrentIndex(index)}
            aria-label={`Go to sample ${index + 1}`}
          />
        ))}
      </div>
    </div>
  );
}
