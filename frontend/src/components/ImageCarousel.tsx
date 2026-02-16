import { useState } from "react";
import styles from "../styles/VideoCarousel.module.css";

interface ImageItem {
  src: string;
  title: string;
  description: string;
}

interface ImageCarouselProps {
  images: ImageItem[];
  compact?: boolean;
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

export function ImageCarousel({ images, compact = false }: ImageCarouselProps) {
  const [currentIndex, setCurrentIndex] = useState(0);

  function goToPrevious() {
    setCurrentIndex((prev) => (prev === 0 ? images.length - 1 : prev - 1));
  }

  function goToNext() {
    setCurrentIndex((prev) => (prev === images.length - 1 ? 0 : prev + 1));
  }

  const currentImage = images[currentIndex];
  const carouselClass = compact
    ? `${styles.carousel} ${styles.carouselCompact}`
    : styles.carousel;

  return (
    <div className={carouselClass}>
      <div className={styles.videoContainer}>
        <button
          className={`${styles.navButton} ${styles.navButtonLeft}`}
          onClick={goToPrevious}
          aria-label="Previous image"
        >
          <ArrowIcon direction="left" />
        </button>

        <div className={styles.videoWrapper}>
          <img
            key={currentImage.src}
            className={styles.video}
            src={currentImage.src}
            alt={currentImage.title}
          />
        </div>

        <button
          className={`${styles.navButton} ${styles.navButtonRight}`}
          onClick={goToNext}
          aria-label="Next image"
        >
          <ArrowIcon direction="right" />
        </button>
      </div>

      <div className={styles.info}>
        <h3 className={styles.videoTitle}>{currentImage.title}</h3>
        <p className={styles.videoDescription}>{currentImage.description}</p>
      </div>

      <div className={styles.indicators}>
        {images.map((_, index) => (
          <button
            key={index}
            className={`${styles.indicator} ${index === currentIndex ? styles.indicatorActive : ""}`}
            onClick={() => setCurrentIndex(index)}
            aria-label={`Go to image ${index + 1}`}
          />
        ))}
      </div>
    </div>
  );
}
