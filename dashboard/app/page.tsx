import { DashboardMetrics } from "@/components/dashboard/dashboard-metrics";
import { RecentActivity } from "@/components/dashboard/recent-activity";
import { DashboardHeader } from "@/components/dashboard/dashboard-header";
import { Suspense } from "react";
import { DashboardSkeleton } from "@/components/dashboard/dashboard-skeleton";

export default function Home() {
  return (
    <div className="flex flex-col gap-6">
      <DashboardHeader />
      <Suspense fallback={<DashboardSkeleton />}>
        <DashboardMetrics />
        <RecentActivity />
      </Suspense>
    </div>
  );
}

