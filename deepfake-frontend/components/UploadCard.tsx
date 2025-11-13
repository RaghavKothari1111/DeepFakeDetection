import { useState, useEffect } from "react";
import { uploadImage, getAuthToken } from "../src/lib/api";
import { useRouter } from "next/router";
import { useCallback } from "react";
import Link from "next/link";

export default function UploadCard() {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const token = getAuthToken();
    setIsAuthenticated(!!token);
  }, []);

  const validate = (f: File) => {
    const allowed = ["image/jpeg", "image/png", "image/jpg"];
    if (!allowed.includes(f.type)) return "Only JPEG and PNG allowed";
    if (f.size > 10_000_000) return "Max file size is 10MB";
    return null;
  };

  const onFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const err = validate(f);
    if (err) {
      setError(err);
      return;
    }
    setError(null);
    setFile(f);
  }, []);

  const onDrop = useCallback(
    (ev: React.DragEvent<HTMLDivElement>) => {
      ev.preventDefault();
      const f = ev.dataTransfer.files?.[0];
      if (!f) return;
      const err = validate(f);
      if (err) {
        setError(err);
        return;
      }
      setError(null);
      setFile(f);
    },
    []
  );

  const onDragOver = useCallback((ev: React.DragEvent<HTMLDivElement>) => {
    ev.preventDefault();
  }, []);

  const onUpload = async () => {
    if (!file) {
      setError("Select a file first");
      return;
    }

    if (!isAuthenticated) {
      setError("Please log in to upload images");
      return;
    }

    setError(null);
    setLoading(true);
    try {
      const res = await uploadImage(file);
      router.push(`/result/${res.jobId}`);
    } catch (e: any) {
      const errorMsg = e?.message || "Upload failed";
      setError(errorMsg);
      console.error("Upload error:", e);
      
      // If authentication error, redirect to login
      if (errorMsg.includes("log in") || errorMsg.includes("authenticated")) {
        setTimeout(() => {
          router.push("/login");
        }, 2000);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded shadow p-6">
      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        className="border-2 border-dashed border-slate-200 rounded p-6 text-center cursor-pointer"
      >
        <p className="text-slate-700 mb-4">Drop an image here or</p>

        <label className="inline-flex items-center gap-2 bg-indigo-600 text-white px-4 py-2 rounded cursor-pointer">
          <input
            onChange={onFileChange}
            type="file"
            accept="image/*"
            className="hidden"
            aria-label="Upload an image"
          />
          Choose file
        </label>

        <div className="mt-3">
          {file && (
            <div className="flex items-center justify-center gap-4">
              <div className="text-sm text-slate-700">{file.name}</div>
              <div className="text-xs text-slate-500">{(file.size / 1024 / 1024).toFixed(2)} MB</div>
              <button
                onClick={() => setFile(null)}
                className="text-sm text-red-600 underline ml-2"
                aria-label="Remove file"
              >
                Remove
              </button>
            </div>
          )}
        </div>

        {error && (
          <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded">
            <p className="text-sm text-red-600">{error}</p>
            {error.includes("log in") && (
              <Link href="/login" className="text-sm text-indigo-600 underline mt-2 inline-block">
                Go to Login →
              </Link>
            )}
          </div>
        )}

        {!isAuthenticated && (
          <div className="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded">
            <p className="text-sm text-yellow-800">
              Please <Link href="/login" className="text-indigo-600 underline">log in</Link> to upload and analyze images.
            </p>
          </div>
        )}

        <div className="mt-6 flex justify-center gap-2">
          <button
            onClick={onUpload}
            className="px-4 py-2 bg-indigo-600 text-white rounded disabled:opacity-60 disabled:cursor-not-allowed"
            disabled={loading || !isAuthenticated || !file}
          >
            {loading ? "Analyzing…" : "Upload & Analyze"}
          </button>
        </div>

        <div className="mt-4 text-xs text-slate-400">
          Supported: JPG, PNG — max 10MB. By uploading you agree to our terms.
        </div>
      </div>
    </div>
  );
}
