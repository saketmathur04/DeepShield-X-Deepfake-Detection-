import { createFileRoute } from "@tanstack/react-router";
import { OverviewPage } from "@/components/deepshield-app";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Overview · DeepShield-X" },
      { name: "description", content: "Review measured deepfake detection experiments across performance, robustness, and explainability." },
      { property: "og:title", content: "Overview · DeepShield-X" },
      { property: "og:description", content: "Review measured deepfake detection experiments across performance, robustness, and explainability." },
    ],
  }),
  component: OverviewPage,
});
