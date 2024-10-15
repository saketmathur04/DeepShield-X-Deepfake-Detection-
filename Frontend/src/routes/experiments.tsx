import { Outlet, createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/experiments")({
  component: () => <Outlet />,
});
