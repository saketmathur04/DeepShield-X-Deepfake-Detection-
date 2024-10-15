/// <reference types="vite/client" />

declare module "@tanstack/react-router" {
  export const Link: any;
  export const useLocation: any;
  export const useRouter: any;
  export const useParams: any;
  export const useNavigate: any;
  export const Outlet: any;
  export const createFileRoute: any;
  export const createRouter: any;
  export const createRootRouteWithContext: <T = any>(...args: any[]) => any;
  export const HeadContent: any;
  export const Scripts: any;
  const content: any;
  export default content;
}

declare module "@tanstack/react-query" {
  export class QueryClient {
    [key: string]: any;
  }
  export const QueryClientProvider: any;
  export const useQuery: any;
  export const useMutation: any;
  const content: any;
  export default content;
}
