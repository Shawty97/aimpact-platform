import { RegisterForm } from "@/components/auth/register-form";
import { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Register | AImpact Platform",
  description: "Create a new AImpact account",
};

export default function RegisterPage() {
  return (
    <div className="flex h-screen w-screen flex-col items-center justify-center

