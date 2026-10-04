---
title: Privacy Policy
---

# Privacy Policy — yt-factory

**Last updated: 4 October 2026**

## What this is

yt-factory is a private, automated publishing tool that its owner runs for a single YouTube
channel of their own. It is not a product, not a service, and not offered to anyone else. It
has no sign-up, no login screen and no user interface. The source code is public so that what
it does can be checked; the running instance belongs to one person.

## Who uses it

One person: the owner of the YouTube channel it publishes to. There are no other users, and
no mechanism by which another person could become one. Nothing in this policy is a promise to
third parties about their data, because the application never receives any.

## What it collects

**From other people: nothing.** The application has no interface through which anyone could
submit information, collects no analytics, sets no cookies, serves no pages, and has no
visitors. This page itself is static and sets no cookies.

**From the owner's own Google account**, after the owner grants access in a browser: an OAuth
refresh token limited to the single scope
`https://www.googleapis.com/auth/youtube.upload`. That scope allows uploading a video and
setting its thumbnail on the owner's channel. It does not allow reading, editing or deleting
anything on the channel, and it gives no access to the owner's identity, e-mail address,
contacts, viewing history, or any other Google service.

The token is stored in two places: on the owner's own computer, and as an encrypted GitHub
Actions secret in the owner's repository. It is never written to the repository, never
printed in logs, and never sent anywhere except to Google.

## What it stores

A SQLite database in the repository, holding only what the application itself produced: the
topics and scripts it wrote, the scene text, the identifiers and scheduled publication times
of the videos it uploaded to the owner's channel, and a log of its own runs. No personal data
of any person is stored in it, and this has been checked before each publication.

## What it does with YouTube data

It calls exactly two YouTube Data API endpoints, `videos.insert` and `thumbnails.set`, and
only against the channel that authorised it. It reads no YouTube data, stores none beyond the
video identifiers of its own uploads, and passes none to anyone. It does not sell data, does
not use it for advertising, and does not use it to train any model. It does not simulate a
viewer and never generates views, likes, subscriptions or comments.

## Content

Videos are produced automatically. The narration is synthetic speech and the artwork is
generated, and every upload declares this to YouTube with
`status.containsSyntheticMedia = true`. Every upload also declares
`status.selfDeclaredMadeForKids = false`. Background music is licensed CC0, with the source of
each track recorded in the repository.

## Third parties

To do its work the application sends text to a language model provider and an image
generation provider, and sends the script's text to a text-to-speech service. What is sent is
the content the application itself wrote for the video. No personal data is sent, because it
holds none.

## Required disclosures for YouTube API clients

- This application uses YouTube API Services.
- By using it, the owner agrees to the
  [YouTube Terms of Service](https://www.youtube.com/t/terms).
- Google's privacy policy is at
  [https://policies.google.com/privacy](https://policies.google.com/privacy), and it explains
  how Google handles data.
- The owner can revoke this application's access to their Google account at any time from the
  [Google security settings page](https://myaccount.google.com/permissions). Revoking access
  stops the application immediately and permanently.

## Retention and deletion

The stored data is the application's own records of what it produced. The owner can delete all
of it at any time by deleting the repository and the local copy, which leaves nothing behind
except the videos already published on the channel.

## Changes

Any change to this policy will be a commit in the public repository, so its full history can
be read at
[github.com/beautyapp800-ctrl/yt-factory-daily](https://github.com/beautyapp800-ctrl/yt-factory-daily).

## Contact

Through the repository's issue tracker:
[github.com/beautyapp800-ctrl/yt-factory-daily/issues](https://github.com/beautyapp800-ctrl/yt-factory-daily/issues)
