import { createFileRoute } from "@tanstack/react-router";
import { AboutPage } from "@/components/deepshield-app";

export const Route = createFileRoute("/about")({
  head: () => ({
    meta: [
      { title: "About DeepShield-X" },
      { name: "description", content: "Learn about the DeepShield-X research objective, dataset labels, and evaluation workflow." },
      { property: "og:title", content: "About DeepShield-X" },
      { property: "og:description", content: "Learn about the DeepShield-X research objective, dataset labels, and evaluation workflow." },
    ],
  }),
  component: AboutPage,
});