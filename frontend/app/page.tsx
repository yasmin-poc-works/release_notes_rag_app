"use client";

import { FormEvent, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Source = { source: string; page: number };
type Message = { role: "user" | "assistant"; text: string; sources?: Source[] };

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function send(event: FormEvent) {
    event.preventDefault();
    if (!input.trim() || loading) return;

    const question = input.trim();
    setInput("");
    setMessages((current) => [...current, { role: "user", text: question }]);
    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Request failed");
      setMessages((current) => [...current, { role: "assistant", text: data.answer, sources: data.sources ?? [] }]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "assistant", text: error instanceof Error ? error.message : "Something went wrong." },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <section className="card">
        <header>
          <div>
            <p className="eyebrow">RELEASE NOTES</p>
            <h1>Ask the changelog</h1>
            <p className="sub">Search product updates with hybrid retrieval.</p>
          </div>
        </header>
        <div className="messages">
          {messages.length === 0 && <div className="empty">Ask about a feature, fix, or release date.</div>}
          {messages.map((message, index) => (
            <div className={`message ${message.role}`} key={index}>
              <span>{message.role === "user" ? "You" : "RAG"}</span>
              {message.role === "assistant" ? (
                <>
                  <div className="answer">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {message.text.replace(/\\([*_])/g, "$1")}
                    </ReactMarkdown>
                  </div>
                  {!!message.sources?.length && (
                    <div className="sources" aria-label="Sources">
                      <strong>Sources</strong>
                      <ul>
                        {message.sources.map((source, sourceIndex) => (
                          <li key={`${source.source}-${source.page}-${sourceIndex}`}>
                            <div>
                              <b>{source.source}</b>
                              <em>Page {source.page}</em>
                            </div>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </>
              ) : (
                <p>{message.text}</p>
              )}
            </div>
          ))}
          {loading && (
            <div className="message assistant">
              <span>RAG</span>
              <p>Searching release notes...</p>
            </div>
          )}
        </div>
        <form onSubmit={send}>
          <input value={input} onChange={(event) => setInput(event.target.value)} placeholder="What changed in the latest release?" />
          <button>Send</button>
        </form>
      </section>
    </main>
  );
}
