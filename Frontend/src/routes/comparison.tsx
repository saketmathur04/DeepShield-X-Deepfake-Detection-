import { createFileRoute } from "@tanstack/react-router";
import { ComparisonPage } from "@/components/deepshield-app";

export const Route = createFileRoute("/comparison")({
  head: () => ({
    meta: [
      { title: "Architecture Benchmark · DeepShield-X" },
      { name: "description", content: "Compare CNN and Vision Transformer architectures using measured DeepShield-X evaluation data." },
      { property: "og:title", content: "Architecture Benchmark · DeepShield-X" },
      { property: "og:description", content: "Compare CNN and Vision Transformer architectures using measured DeepShield-X evaluation data." },
    ],
  }),
  component: ComparisonPage,
});