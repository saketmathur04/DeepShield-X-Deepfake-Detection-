import { createFileRoute } from "@tanstack/react-router";
import { ExplainabilityPage } from "@/components/deepshield-app";

export const Route = createFileRoute("/explainability")({
  head: () => ({
    meta: [
      { title: "Model Explainability · DeepShield-X" },
      { name: "description", content: "Review the DeepShield-X evidence layout for model attribution and Grad-CAM analysis." },
      { property: "og:title", content: "Model Explainability · DeepShield-X" },
      { property: "og:description", content: "Review the DeepShield-X evidence layout for model attribution and Grad-CAM analysis." },
    ],
  }),
  component: ExplainabilityPage,
});