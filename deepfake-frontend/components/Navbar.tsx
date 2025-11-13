/* navbar */import Link from "next/link";
import { useEffect, useState } from "react";
import { getAuthToken, logout } from "../src/lib/api";
import { useRouter } from "next/router";

export default function Navbar() {
  const [signedIn, setSignedIn] = useState(false);
  const router = useRouter();

  useEffect(() => {
    // Check if user is authenticated
    const token = getAuthToken();
    setSignedIn(!!token);
  }, []);

  const handleLogout = () => {
    logout();
    setSignedIn(false);
    router.push("/");
  };

  return (
    <header className="bg-white border-b border-slate-200 shadow-sm sticky top-0 z-10">
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
        {/* Logo left */}
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 bg-indigo-600 text-white font-bold rounded-md flex items-center justify-center">
            D
          </div>
          <Link href="/">
            <span className="font-semibold text-slate-800 cursor-pointer">
              DeepVerify
            </span>
          </Link>
        </div>

        <nav className="flex items-center gap-4 text-slate-700">
          {!signedIn ? (
            <>
              <Link href="/login">
                <span className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 cursor-pointer inline-block">
              Sign In
                </span>
              </Link>
              <Link href="/register">
                <span className="px-4 py-2 bg-slate-100 text-slate-700 rounded hover:bg-slate-200 cursor-pointer inline-block">
                  Sign Up
                </span>
              </Link>
            </>
          ) : (
            <>
              <Link href="/dashboard">
                <span className="hover:text-indigo-600 cursor-pointer">
                  Dashboard
                </span>
              </Link>

              <Link href="/membership">
                <span className="hover:text-indigo-600 cursor-pointer">
                  Membership
                </span>
              </Link>

              <Link href="/support">
                <span className="hover:text-indigo-600 cursor-pointer">
                  Support
                </span>
              </Link>

              <button
                onClick={handleLogout}
                className="px-3 py-2 bg-red-50 text-red-700 rounded hover:bg-red-100"
              >
                Sign Out
              </button>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}
