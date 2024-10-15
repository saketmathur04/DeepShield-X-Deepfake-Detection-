import { createFileRoute } from "@tanstack/react-router";
import { ExperimentsPage } from "@/components/deepshield-app";

export const Route = createFileRoute("/experiments/")({
  head: () => ({
    meta: [
      { title: "Experiments · DeepShield-X" },
      { name: "description", content: "Track reproducible DeepShield-X training runs and inspect measured evaluation artifacts." },
      { property: "og:title", content: "Experiments · DeepShield-X" },
      { property: "og:description", content: "Track reproducible DeepShield-X training runs and inspect measured evaluation artifacts." },
    ],
  }),
  component: ExperimentsPage,
});
