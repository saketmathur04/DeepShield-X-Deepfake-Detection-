import { createFileRoute } from "@tanstack/react-router";
import { ExperimentDetailPage } from "@/components/deepshield-app";

export const Route = createFileRoute("/experiments/$id")({
  head: ({ params }: any) => ({
    meta: [
      { title: `${params.id} · DeepShield-X` },
      { name: "description", content: "Inspect the configuration, measured metrics, and evaluation artifacts for a DeepShield-X experiment." },
      { property: "og:title", content: `${params.id} · DeepShield-X` },
      { property: "og:description", content: "Inspect the configuration, measured metrics, and evaluation artifacts for a DeepShield-X experiment." },
      { name: "robots", content: "noindex" },
    ],
  }),
  component: () => <ExperimentDetailPage id={Route.useParams().id} />,
});