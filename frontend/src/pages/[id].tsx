import Head from "next/head";
import dynamic from "next/dynamic";
import ConsensusCard from "../../src/components/ConsensusCard";
import ModelResultCard from "../../src/components/ModelResultCard";
import useSWR from "swr";
import { fetcher, getAuthHeaders } from "../../src/lib/api";
import { useRouter } from "next/router";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { RefreshCw, Download, Loader2 } from "lucide-react";
import { useState } from "react";
const Navbar = dynamic(() => import("@/components/Navbar"), { ssr: false });

export default function ResultPage() {
  const router = useRouter();
  const { id } = router.query;
  const [isRerunning, setIsRerunning] = useState(false);

  const { data: job, error, mutate } = useSWR(
    () => (id ? `/api/jobs/${id}` : null),
    fetcher,
    { 
      refreshInterval: (data) => {
        // Keep refreshing if job is processing, stop when completed
        if (data?.status === "pending" || data?.status === "processing") {
          return 1000; // Refresh every 1 second when processing
        }
        return 0; // Stop auto-refresh when completed
      },
      revalidateOnFocus: true,
      revalidateOnReconnect: true,
    }
  );

  const isLoading = !job && !error;
  const isProcessing =
    job?.status === "pending" || job?.status === "processing";

  const handleRerun = async () => {
    if (!id || isRerunning) return;
    
    setIsRerunning(true);
    try {
      const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${API_BASE}/api/jobs/${id}/rerun`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeaders(),
        },
        credentials: "include",
      });
      
      if (!response.ok) {
        const error = await response.json();
        alert(`Failed to re-run analysis: ${error.detail || "Unknown error"}`);
        return;
      }
      
      // Immediately refresh the job data to show pending status
      await mutate();
      // Show success message
      alert("Analysis re-run started! The page will update automatically.");
    } catch (err) {
      console.error("Error re-running analysis:", err);
      alert("Failed to re-run analysis. Please try again.");
    } finally {
      setIsRerunning(false);
    }
  };

  // ---------- LOADING UI ----------
  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />

        <main className="pt-16 max-w-6xl mx-auto px-4 py-12">
          <div className="space-y-6">
            <Skeleton className="h-8 w-1/3" />
            <Skeleton className="h-64 w-full" />
            <Skeleton className="h-64 w-full" />
          </div>
        </main>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="pt-16 max-w-4xl mx-auto px-4 py-12 text-center">
          <p className="text-muted-foreground">Job not found.</p>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Head>
        <title>Analysis Result — {job.job_id}</title>
      </Head>

      <Navbar />

      <main className="pt-16 max-w-7xl mx-auto px-4 py-12">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Analysis Results</h1>
          <p className="text-muted-foreground">{job.job_id}</p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* LEFT COLUMN — CONSENSUS + MODELS */}
          <div className="lg:col-span-2 space-y-10">
            {/* Consensus Card */}
            <ConsensusCard
              consensus={job.consensus}
              imageUrl={job.image?.thumbnail_url}
            />

            {/* Per-model breakdown */}
            <section>
              <h3 className="text-2xl font-semibold mb-5">
                Per-Model Breakdown
              </h3>

              <div className="grid md:grid-cols-2 gap-6">
                {job.models?.map((model: any, index: number) => (
                  <ModelResultCard key={index} model={model} />
                ))}
              </div>
            </section>
          </div>

          {/* RIGHT SIDEBAR */}
          <aside className="space-y-6 sticky top-24 h-fit">
            {/* Image */}
            <Card className="p-3">
              <img
                src={job.image?.thumbnail_url}
                alt="Analyzed Image"
                className="rounded-md w-full"
              />
            </Card>

            {/* Buttons */}
            <div className="space-y-3">
              <Button 
                variant="outline" 
                className="w-full"
                onClick={handleRerun}
                disabled={isRerunning || isProcessing}
              >
                {isRerunning ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Re-running...
                  </>
                ) : (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Re-run Analysis
                  </>
                )}
              </Button>

              <Button variant="outline" className="w-full">
                <Download className="h-4 w-4 mr-2" />
                Download PDF
              </Button>
            </div>

            {/* Details */}
            <Card className="p-5">
              <h4 className="font-semibold mb-1">Details</h4>
              <p className="text-sm text-muted-foreground">
                Uploaded: {new Date(job.created_at).toLocaleString()}
              </p>
            </Card>
          </aside>
        </div>
      </main>
    </div>
  );
}
