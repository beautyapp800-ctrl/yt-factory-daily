# M1 Stories - channel description, keywords and art

Channel: **M1 Stories**, @M1Stories-g5j. The name and the handle are unchanged.

For YouTube Studio: Customisation > Basic info (description) and Settings > Channel >
Basic info (keywords). The name and the handle are unchanged. Both cover the money
videos and the Stoic ones through the one thing they share - character and discipline.

## Description (992 of 1000 characters)

```
Fortunes and philosophies look like different subjects. They are not. Behind a first million and a page of Marcus Aurelius sits the same short list: refusing what does not matter, staying steady when it would be easier not to, and repeating dull work long after the excitement has gone.

This channel tells both halves of that story.

One half is biography. How particular people actually built their first million - founders, traders, outsiders - told as mechanics rather than myth: what they refused, what they repeated, what it cost.

The other half is Stoic philosophy, told the same way. Long, calm videos on self-discipline, self-control, judgement and the difference between what is in your hands and what is not. Seneca, Epictetus and Marcus Aurelius wrote for people with work to do.

Either way the subject is character - the part of success nobody can hand you, and the part that holds whether the money arrives or it does not.

New long-form videos regularly. Best watched slowly.
```

## Keywords (414 of 500 characters, 24 terms)

```
stoicism, stoic philosophy, marcus aurelius, seneca, epictetus, self discipline, self control, mental toughness, emotional resilience, how to build wealth, first million, self made millionaire, success stories, entrepreneur mindset, money mindset, financial discipline, delayed gratification, personal growth, character building, philosophy for everyday life, stoic lessons, discipline motivation, calm mind, focus
```

## Avatar and banner

`docs/channel/avatar.png` (800x800) and `docs/channel/banner.png` (2560x1440), drawn by
`scripts/channel_art.py`:

    python scripts/channel_art.py --name "M1 Stories" --monogram "M1"

The banner reads `M1 STORIES` over `FIRST MILLIONS AND STOIC DISCIPLINE`, which names both
halves of the channel in the order a new viewer meets them. The avatar keeps the existing
`M1` mark rather than inventing a second one - the channel is five months old and that mark
is what its audience already recognises - and restates it in the dark palette the new
thumbnails use.

Both are built from the thumbnails' own language: Anton, white over amber (240, 176, 84),
near-black with a warm light low and left. The banner's backdrop is made from that palette
rather than taken from a video frame, because every candidate frame carried a sign with
invented lettering, a screen or a sheet of paper - the things the picture rules exist to
keep out - and the banner is the one image on the channel that is never redrawn.

Checked rather than assumed: the two lines occupy 345px of the 423px that every device
shows, so no crop can cut them; the darkening spans the full 2560px, so the desktop strip
finds no bright edge; and the `M1` is 23.8px tall when the avatar is drawn at the 48px it is
actually read at, against a floor of 13px.
