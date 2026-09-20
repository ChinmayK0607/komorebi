# Content identity repair diagnostic

This run tests whether a content-addressed asset table and explicit role
aliases prevent the duplicate-image failure seen in the earlier validation.
It contains the six existing real pairs in both candidate orientations. The
two public shards contain one opaque view per real pair each; \`bindings.json\`
keeps the orientation and source IDs private for audit.

\`assets.json\` has one entry per unique PNG byte hash. Each public item in
\`judge-01.json\` or \`judge-02.json\` contains:

* \`rolemap\`: the opaque asset ID bound to \`reference\`, \`A\`, and \`B\`;
* \`asset_ids\`: those IDs in first-role order with duplicates removed; and
* \`display_order\`: always \`reference\`, \`A\`, \`B\`.

Resolve and verify a shard before display:

\`\`\`sh
python3 prepare.py import --assets assets.json --manifest judge-01.json
python3 prepare.py resolve --assets assets.json --manifest judge-01.json > judge-01-resolved.json
\`\`\`

For each item, display every entry in its \`display\` list exactly once, using
the returned path. Then apply \`rolemap\` to associate the saved observations
with the three roles. For example, if \`display\` contains \`asset-004\` once and
\`rolemap["A"] == "asset-004"\`, that same observation is the observation for A.
Persist the item before loading the next one. A \`displayed_asset_ids\` list in a
label is useful bookkeeping, but it is a self-report and does not prove that
the image was displayed; the display calls and saved observations remain the
inspection record.

Use the existing Luna low judgment fields (\`winner\`, \`confidence\`,
\`acceptable_A\`, \`acceptable_B\`, \`reason\`, and role-keyed \`observations\`) and
add \`rolemap\` plus \`displayed_asset_ids\`. Do not add image paths to labels.
The opaque item ID and ordinal must be preserved exactly.

\`deterministic-controls.json\` contains the six original controls. They are
excluded from both visual judge shards. Exact candidate identity resolves a
relative winner (including a tie for byte-identical A/B); an exact-reference
match is labelled separately. These outcomes are deterministic control
results, not visual-judge successes and must not be converted into
acceptability claims or training rewards.

Run the focused checks with:

\`\`\`sh
python3 -m unittest -v test_prepare.py
\`\`\`
