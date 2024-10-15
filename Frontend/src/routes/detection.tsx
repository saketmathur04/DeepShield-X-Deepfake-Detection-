import { createFileRoute } from "@tanstack/react-router";
import { DetectionPage } from "@/components/deepshield-app";

export const Route = createFileRoute("/detection")({
  head: () => ({
    meta: [
      { title: "Detection · DeepShield-X" },
      { name: "description", content: "Run an illustrative image or video deepfake detection workflow in the DeepShield-X research console." },
      { property: "og:title", content: "Detection · DeepShield-X" },
      { property: "og:description", content: "Run an illustrative image or video deepfake detection workflow in the DeepShield-X research console." },
    ],
  }),
  component: DetectionPage,
});