import { useState } from "react";
import styles from "../styles/VideoCarousel.module.css";

interface VideoItem {
  src: string;
  title: string;
  description: string;
}

interface VideoCarouselProps {
  videos: VideoItem[];
  hideText?: boolean;
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

export function VideoCarousel({ videos, hideText = false }: VideoCarouselProps) {
  const [currentIndex, setCurrentIndex] = useState(0);

  function goToPrevious() {
    setCurrentIndex((prev) => (prev === 0 ? videos.length - 1 : prev - 1));
  }

  function goToNext() {
    setCurrentIndex((prev) => (prev === videos.length - 1 ? 0 : prev + 1));
  }

  const currentVideo = videos[currentIndex];

  return (
    <div className={styles.carousel}>
      <div className={styles.videoContainer}>
        <button
          className={`${styles.navButton} ${styles.navButtonLeft}`}
          onClick={goToPrevious}
          aria-label="Previous video"
        >
          <ArrowIcon direction="left" />
        </button>

        <div className={styles.videoWrapper}>
          <video
            key={currentVideo.src}
            className={styles.video}
            controls
            autoPlay
            loop
            muted
          >
            <source src={currentVideo.src} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>

        <button
          className={`${styles.navButton} ${styles.navButtonRight}`}
          onClick={goToNext}
          aria-label="Next video"
        >
          <ArrowIcon direction="right" />
        </button>
      </div>

      {!hideText && (
        <div className={styles.info}>
          <h3 className={styles.videoTitle}>{currentVideo.title}</h3>
          <p className={styles.videoDescription}>{currentVideo.description}</p>
        </div>
      )}

      <div className={styles.indicators}>
        {videos.map((_, index) => (
          <button
            key={index}
            className={`${styles.indicator} ${index === currentIndex ? styles.indicatorActive : ""}`}
            onClick={() => setCurrentIndex(index)}
            aria-label={`Go to video ${index + 1}`}
          />
        ))}
      </div>
    </div>
  );
}
