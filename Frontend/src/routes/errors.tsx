import { createFileRoute } from "@tanstack/react-router";
import { ErrorsPage } from "@/components/deepshield-app";

export const Route = createFileRoute("/errors")({
  head: () => ({
    meta: [
      { title: "Error Analysis · DeepShield-X" },
      { name: "description", content: "Inspect confidence, label agreement, and high-confidence failure modes in deepfake detection." },
      { property: "og:title", content: "Error Analysis · DeepShield-X" },
      { property: "og:description", content: "Inspect confidence, label agreement, and high-confidence failure modes in deepfake detection." },
    ],
  }),
  component: ErrorsPage,
});