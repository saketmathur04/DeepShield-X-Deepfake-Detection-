import { createFileRoute } from "@tanstack/react-router";
import { RobustnessPage } from "@/components/deepshield-app";

export const Route = createFileRoute("/robustness")({
  head: () => ({
    meta: [
      { title: "Robustness Analysis · DeepShield-X" },
      { name: "description", content: "Evaluate how deepfake detection performance changes under image degradation conditions." },
      { property: "og:title", content: "Robustness Analysis · DeepShield-X" },
      { property: "og:description", content: "Evaluate how deepfake detection performance changes under image degradation conditions." },
    ],
  }),
  component: RobustnessPage,
});