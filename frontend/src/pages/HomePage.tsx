import { useEffect, useState } from "react";
import styles from "../styles/HomePage.module.css";
import { VideoCarousel } from "../components/VideoCarousel";
import { ImageCarousel } from "../components/ImageCarousel";

const sections = [
  { id: "hero", title: "Introduction" },
  { id: "history", title: "History" },
  { id: "1d-example", title: "1D Example" },
  { id: "2d-example", title: "2D Example" },
  { id: "3d-example", title: "3D Example" },
  { id: "results", title: "Results" },
  { id: "conclusion", title: "Conclusion" },
];

function ChevronIcon({ direction }: { direction: "left" | "right" }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{
        transform: direction === "left" ? "rotate(180deg)" : "none",
      }}
    >
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}

function TimelineNav({
  activeSection,
  collapsed,
  onToggle,
}: {
  activeSection: string;
  collapsed: boolean;
  onToggle: () => void;
}) {
  return (
    <nav
      className={`${styles.timelineNav} ${collapsed ? styles.timelineNavCollapsed : ""}`}
    >
      <ul className={styles.timelineList}>
        {sections.map((section) => (
          <li key={section.id} className={styles.timelineItem}>
            <a
              href={`#${section.id}`}
              className={`${styles.timelineLink} ${
                activeSection === section.id ? styles.timelineLinkActive : ""
              }`}
            >
              <span className={styles.timelineDot} />
              <span className={styles.timelineLabel}>{section.title}</span>
            </a>
          </li>
        ))}
      </ul>
      <button
        className={styles.timelineToggle}
        onClick={onToggle}
        aria-label={collapsed ? "Expand navigation" : "Collapse navigation"}
      >
        <ChevronIcon direction={collapsed ? "right" : "left"} />
      </button>
    </nav>
  );
}

type NoiseLevel = "none" | "5" | "10";

function NoiseSelector({
  value,
  onChange,
}: {
  value: NoiseLevel;
  onChange: (level: NoiseLevel) => void;
}) {
  const options: { level: NoiseLevel; label: string }[] = [
    { level: "none", label: "No Noise" },
    { level: "5", label: "5% Noise" },
    { level: "10", label: "10% Noise" },
  ];

  return (
    <div className={styles.noiseSelector}>
      {options.map((option) => (
        <button
          key={option.level}
          className={`${styles.noiseSelectorButton} ${
            value === option.level ? styles.noiseSelectorButtonActive : ""
          }`}
          onClick={() => onChange(option.level)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function get1DResultImages(noiseLevel: NoiseLevel) {
  const prefix =
    noiseLevel === "none"
      ? "1d-inverse_model_results"
      : `1d-${noiseLevel}noise_inverse_model_results`;

  const noiseDescription =
    noiseLevel === "none"
      ? ""
      : ` Training data includes ${noiseLevel}% Gaussian noise added to sensor measurements.`;

  return [
    {
      src: `/images/${prefix}_01.png`,
      title: "Test Set Results (1/4)",
      description: `Comparison of ground truth density profiles (blue) with neural network predictions (gray). Each pair shows the true inclusion position and density above the network's prediction.${noiseDescription}`,
    },
    {
      src: `/images/${prefix}_02.png`,
      title: "Test Set Results (2/4)",
      description: `The neural network learns to map sensor measurements to 1D density profiles, predicting both the location and density of inclusions within the tunnel.${noiseDescription}`,
    },
    {
      src: `/images/${prefix}_03.png`,
      title: "Test Set Results (3/4)",
      description: `These test samples were held out during training. The model generalizes to unseen inclusion configurations with varying positions, sizes, and densities.${noiseDescription}`,
    },
    {
      src: `/images/${prefix}_04.png`,
      title: "Test Set Results (4/4)",
      description: `Final test samples demonstrating the inverse model's ability to reconstruct density profiles from boundary sensor measurements alone.${noiseDescription}`,
    },
  ];
}

function get2DResultImages(noiseLevel: NoiseLevel) {
  const prefix =
    noiseLevel === "none"
      ? "2d-inverse_model_results"
      : `2d-${noiseLevel}noise_inverse_model_results`;

  const noiseDescription =
    noiseLevel === "none"
      ? ""
      : ` Training data includes ${noiseLevel}% Gaussian noise added to sensor measurements.`;

  // 5% noise only has 6 images
  const imageCount = noiseLevel === "5" ? 6 : 7;

  const descriptions = [
    "Comparison of ground truth inclusion outlines (left) with neural network predictions (right). The network learns to reconstruct inclusion geometry from sensor measurements alone.",
    "The neural network predicts k-space coefficients which are transformed back to real space via inverse FFT, producing these reconstructed inclusion images.",
    "Each pair shows the true inclusion shape and position alongside the network's prediction, demonstrating the model's ability to localize inclusions.",
    "The model handles all three inclusion types (circle, square, triangle) and varying positions within the domain.",
    "These test samples were held out during training, demonstrating the model's generalization to unseen configurations.",
    "The inverse problem maps 61,840 sensor measurements to 8,192 k-space coefficients representing a 64x64 spatial grid.",
    "Final test samples showing the neural network's reconstruction quality across different inclusion types and positions.",
  ];

  return Array.from({ length: imageCount }, (_, i) => ({
    src: `/images/${prefix}_0${i + 1}.png`,
    title: `Test Set Results (${i + 1}/${imageCount})`,
    description: `${descriptions[i]}${noiseDescription}`,
  }));
}

export function HomePage() {
  const [activeSection, setActiveSection] = useState("hero");
  const [navCollapsed, setNavCollapsed] = useState(true);
  const [noiseLevel1D, setNoiseLevel1D] = useState<NoiseLevel>("none");
  const [noiseLevel2D, setNoiseLevel2D] = useState<NoiseLevel>("none");

  function handleGetStarted() {
    setNavCollapsed(false);
    document.getElementById("history")?.scrollIntoView({ behavior: "smooth" });
  }

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      { rootMargin: "-50% 0px -50% 0px" }
    );

    sections.forEach((section) => {
      const element = document.getElementById(section.id);
      if (element) {
        observer.observe(element);
      }
    });

    return () => observer.disconnect();
  }, []);

  return (
    <div className={styles.pageContainer}>
      <TimelineNav
        activeSection={activeSection}
        collapsed={navCollapsed}
        onToggle={() => setNavCollapsed(!navCollapsed)}
      />

      <div className={styles.content}>
        <section id="hero" className={styles.hero}>
          <h1 className={styles.title}>Simulation Based Imaging</h1>
          <p className={styles.subtitle}>
            An interactive exploration of simulation-based imaging techniques
          </p>
          <div className={styles.heroCarousel}>
            <VideoCarousel
              hideText
              videos={[
                {
                  src: "/videos/3d_video_1.mp4",
                  title: "Multiple Sources (Isosurfaces)",
                  description: "",
                },
                {
                  src: "/videos/3d_video_2.mp4",
                  title: "Multiple Sources (Points)",
                  description: "",
                },
                {
                  src: "/videos/3d_video_3.mp4",
                  title: "Single Source (Points)",
                  description: "",
                },
              ]}
            />
          </div>
          <button className={styles.getStartedButton} onClick={handleGetStarted}>
            Get Started
          </button>
        </section>

        <section id="history" className={styles.section}>
          <h2 className={styles.sectionTitle}>History</h2>
          <div className={styles.sectionContent}></div>
        </section>

        <section id="1d-example" className={styles.section}>
          <h2 className={styles.sectionTitle}>1D Example</h2>
          <div className={styles.sectionContent}>
            <VideoCarousel
              videos={[
                {
                  src: "/videos/1d_center_inclusion.mp4",
                  title: "Center Inclusion",
                  description:
                    "A 1D acoustic pulse propagates through a tunnel with a central inclusion. The inclusion has doubled density and wave speed, causing partial reflection and transmission of the wave.",
                },
                {
                  src: "/videos/1d_dense_inclusion.mp4",
                  title: "Dense Left Inclusion",
                  description:
                    "A wider inclusion shifted to the left with higher density and wave speed (3x). The increased impedance contrast produces stronger reflections.",
                },
                {
                  src: "/videos/1d_right_inclusion.mp4",
                  title: "Wide Right Inclusion",
                  description:
                    "A wide inclusion (0.25) positioned to the right with density and wave speed of 3.5x. The wave travels further before encountering the inclusion.",
                },
              ]}
            />
          </div>

          <div className={styles.subslide}>
            <h3 className={styles.subslideTitle}>Batch Simulation Results</h3>
            <ImageCarousel
              compact
              images={[
                {
                  src: "/images/1d-inclusion_plots.png",
                  title: "Inclusion Configurations",
                  description:
                    "A grid of 25 random 1D inclusion configurations used for training. Each subplot shows the position and size of the inclusion within the tunnel, with opacity indicating density.",
                },
                {
                  src: "/images/1d-sensor-data-grid.png",
                  title: "Sensor Data",
                  description:
                    "Corresponding sensor measurements for each simulation. The left sensor (red) and right sensor (cyan) record the acoustic wave as it propagates through the tunnel and interacts with the inclusion.",
                },
              ]}
            />
          </div>

          <div className={styles.subslide}>
            <div className={styles.subslideTitleRow}>
              <h3 className={styles.subslideTitle}>Neural Network Results</h3>
              <NoiseSelector value={noiseLevel1D} onChange={setNoiseLevel1D} />
            </div>
            <ImageCarousel compact images={get1DResultImages(noiseLevel1D)} />
          </div>
        </section>

        <section id="2d-example" className={styles.section}>
          <h2 className={styles.sectionTitle}>2D Example</h2>
          <div className={styles.sectionContent}>
            <VideoCarousel
              videos={[
                {
                  src: "/videos/circle_inclusion.mp4",
                  title: "Circle Inclusion",
                  description:
                    "A circular inclusion scatters the incoming acoustic wave. The wavefront diffracts around the inclusion and interferes with reflections from the rigid boundaries.",
                },
                {
                  src: "/videos/triangle_inclusion.mp4",
                  title: "Triangle Inclusion",
                  description:
                    "An equilateral triangular inclusion creates sharp diffraction patterns at its vertices. The flat edges produce distinct reflected wavefronts.",
                },
                {
                  src: "/videos/square_inclusion.mp4",
                  title: "Square Inclusion",
                  description:
                    "A square inclusion positioned off-center demonstrates how corner diffraction and edge reflections combine to produce complex interference patterns.",
                },
              ]}
            />
          </div>

          <div className={styles.subslide}>
            <h3 className={styles.subslideTitle}>Batch Simulation Results</h3>
            <ImageCarousel
              compact
              images={[
                {
                  src: "/images/2d-inclusion_plots.png",
                  title: "Inclusion Configurations",
                  description:
                    "A grid of 25 random inclusion configurations used for training. Each subplot shows the position and shape (circle, triangle, or square) of the inclusion within the domain.",
                },
                {
                  src: "/images/2d-sensor-data-grid.png",
                  title: "Sensor Data",
                  description:
                    "Corresponding sensor measurements for each simulation. The boundary sensors record the acoustic wave as it propagates through the domain and interacts with the inclusion.",
                },
              ]}
            />
          </div>

          <div className={styles.subslide}>
            <div className={styles.subslideTitleRow}>
              <h3 className={styles.subslideTitle}>Neural Network Results</h3>
              <NoiseSelector value={noiseLevel2D} onChange={setNoiseLevel2D} />
            </div>
            <ImageCarousel compact images={get2DResultImages(noiseLevel2D)} />
          </div>
        </section>

        <section id="3d-example" className={styles.section}>
          <h2 className={styles.sectionTitle}>3D Example</h2>
          <div className={styles.sectionContent}>
            <VideoCarousel
              videos={[
                {
                  src: "/videos/3d_video_1.mp4",
                  title: "Multiple Sources (Isosurfaces)",
                  description:
                    "A 3D acoustic simulation with multiple sources. The pressure field is visualized using isosurfaces, showing wave propagation through the tetrahedral mesh domain with an embedded inclusion.",
                },
                {
                  src: "/videos/3d_video_2.mp4",
                  title: "Multiple Sources (Points)",
                  description:
                    "Multiple acoustic sources emit waves that interact with each other and the inclusion. The point cloud rendering shows pressure values at each node of the tetrahedral mesh.",
                },
                {
                  src: "/videos/3d_video_3.mp4",
                  title: "Single Source (Points)",
                  description:
                    "A single acoustic source emits waves that propagate through the domain and interact with the inclusion. The point cloud visualization reveals the wave structure at mesh nodes.",
                },
              ]}
            />
          </div>
        </section>

        <section id="results" className={styles.section}>
          <h2 className={styles.sectionTitle}>Results</h2>
          <div className={styles.sectionContent}></div>
        </section>

        <section id="conclusion" className={styles.section}>
          <h2 className={styles.sectionTitle}>Conclusion</h2>
          <div className={styles.sectionContent}></div>
        </section>
      </div>
    </div>
  );
}
