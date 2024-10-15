import { createFileRoute } from "@tanstack/react-router";
import { ReportsPage } from "@/components/deepshield-app";

export const Route = createFileRoute("/reports")({
  head: () => ({
    meta: [
      { title: "Research Reports · DeepShield-X" },
      { name: "description", content: "Review export-ready DeepShield-X research artifacts and report status." },
      { property: "og:title", content: "Research Reports · DeepShield-X" },
      { property: "og:description", content: "Review export-ready DeepShield-X research artifacts and report status." },
    ],
  }),
  component: ReportsPage,
});