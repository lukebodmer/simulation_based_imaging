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

export function HomePage() {
  const [activeSection, setActiveSection] = useState("hero");
  const [navCollapsed, setNavCollapsed] = useState(true);

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
          <div className={styles.sectionContent}></div>
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
            <h3 className={styles.subslideTitle}>Neural Network Results</h3>
            <ImageCarousel
              compact
              images={[
                {
                  src: "/images/2d-inverse_model_results_01.png",
                  title: "Test Set Results (1/7)",
                  description:
                    "Comparison of ground truth inclusion outlines (left) with neural network predictions (right). The network learns to reconstruct inclusion geometry from sensor measurements alone.",
                },
                {
                  src: "/images/2d-inverse_model_results_02.png",
                  title: "Test Set Results (2/7)",
                  description:
                    "The neural network predicts k-space coefficients which are transformed back to real space via inverse FFT, producing these reconstructed inclusion images.",
                },
                {
                  src: "/images/2d-inverse_model_results_03.png",
                  title: "Test Set Results (3/7)",
                  description:
                    "Each pair shows the true inclusion shape and position alongside the network's prediction, demonstrating the model's ability to localize inclusions.",
                },
                {
                  src: "/images/2d-inverse_model_results_04.png",
                  title: "Test Set Results (4/7)",
                  description:
                    "The model handles all three inclusion types (circle, square, triangle) and varying positions within the domain.",
                },
                {
                  src: "/images/2d-inverse_model_results_05.png",
                  title: "Test Set Results (5/7)",
                  description:
                    "These test samples were held out during training, demonstrating the model's generalization to unseen configurations.",
                },
                {
                  src: "/images/2d-inverse_model_results_06.png",
                  title: "Test Set Results (6/7)",
                  description:
                    "The inverse problem maps 61,840 sensor measurements to 8,192 k-space coefficients representing a 64x64 spatial grid.",
                },
                {
                  src: "/images/2d-inverse_model_results_07.png",
                  title: "Test Set Results (7/7)",
                  description:
                    "Final test samples showing the neural network's reconstruction quality across different inclusion types and positions.",
                },
              ]}
            />
          </div>
        </section>

        <section id="3d-example" className={styles.section}>
          <h2 className={styles.sectionTitle}>3D Example</h2>
          <div className={styles.sectionContent}></div>
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
