PLAN = """Turn this question into search queries for a vector store.

If it asks about several distinct things, emit one query per thing. Each query
must stand alone and name exactly one topic — a query naming two topics
retrieves documents about neither.

"compare X and Y" becomes ["X", "Y"], not ["X vs Y"].

Question: {question}"""

REPLAN = """These queries returned too few relevant documents. Try different wording.

Keep one query per topic. Move toward vocabulary likely to appear in the source
text rather than the asker's phrasing.

Failed queries: {queries}
Question: {question}"""

GRADE = """Judge this document against the search query below.

Grade it against this query alone, not any broader question it came from.

on_topic: does it contain information relevant to this query?
satisfies_constraints: does it meet explicit constraints in the query, such as
dates or named sources? True if the query states none.

Metadata: {metadata}

Document:
{document}

Query: {query}"""

ANSWER = """Answer the question using only the context below.

Give everything the context supports before noting anything it lacks. If the
question asks for a comparison the context doesn't make directly, describe each
side from what's there and say the direct comparison isn't in the source.

Never pad with outside knowledge. But never withhold something the context
does contain because it falls short of the full question.

Context:
{context}

Question: {question}"""

GROUND = """Check whether this answer is supported by the context it was given.

A claim is supported if the context states it or directly implies it. Correct
outside knowledge that isn't in the context counts as unsupported.

An answer that says the context lacks the information is grounded — declining
to answer is not a claim.

Context:
{context}

Answer:
{answer}"""

USEFUL = """Does this answer extract everything the context offers toward the question?

Judge the answer against the context, not against an ideal answer. If the
context only supports a partial response, a partial response is correct —
and saying which part is missing is better still.

Not useful means: the context contained relevant material the answer ignored,
or the answer drifted to a different topic.

Useful means: the answer used what was there, including when what was there
was less than the question asked for.

Question: {question}

Context:
{context}

Answer:
{answer}"""

STRICTER = """Answer the question using only the context below.

A previous attempt included claims absent from the context:
{unsupported}

Restrict yourself to what the context states. Where it's silent, say so.

Context:
{context}

Question: {question}"""