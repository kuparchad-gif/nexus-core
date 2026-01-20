import fs from 'fs'
import path from 'path'

export type PromptVar = { key: string, label?: string, type?: 'string'|'number'|'enum', default?: any, options?: string[] }
export type PromptDef = {
  id: string
  name: string
  tags?: string[]
  system?: string
  developer?: string
  user_template?: string
  variables?: PromptVar[]
  examples?: { input: string, output?: string }[]
}

export type PromptDB = {
  version: string
  prompts: PromptDef[]
}

const DATA_DIR = path.join(process.cwd(), 'data')
const DATA_PATH = path.join(DATA_DIR, 'prompts.json')

export function ensureStore(){
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true })
  if (!fs.existsSync(DATA_PATH)){
    const seed: PromptDB = {
      version: '1.0',
      prompts: [
        {
          id: 'poe-edgar',
          name: 'Edgar Allan',
          tags: ['style','gothic','poetry','edgar-allan-poe'],
          system: [
            "You emulate the literary voice of Edgar Allan Poe (public domain).",
            "Your tone is melancholy, precise, and gothic; imagery is vivid but economical.",
            "Favor internal rhyme, alliteration, and a cadence reminiscent of trochaic meter.",
            "Avoid direct quotation of existing Poe texts; create wholly original prose or verse."
          ].join("\n"),
          developer: [
            "Constraints:",
            "- Aim for 1–3 short stanzas or 2–5 compact paragraphs.",
            "- Prefer archaic diction sparingly (e.g., 'o'er', 'nigh') without overuse.",
            "- End with a poignant turn or chilling resonance."
          ].join("\n"),
          user_template: "Compose a {form} about {topic} with a {mood} mood.",
          variables: [
            { key: "form", label: "Form", type: "enum", default: "short poem", options: ["short poem","lyric prose","monologue"] },
            { key: "topic", label: "Topic", type: "string", default: "a midnight visitor" },
            { key: "mood", label: "Mood", type: "enum", default: "brooding", options: ["brooding","wistful","ominous","tragic"] }
          ],
          examples: [
            { input: "Compose a short poem about a midnight visitor with a brooding mood." }
          ]
        },
        {
          id: 'raven-avatar',
          name: 'The Raven',
          tags: ['persona','poetry','symbolic'],
          system: [
            "You speak as a prophetic raven, terse and oracular.",
            "Lines are short; cadence steady; refrain appears sparingly.",
            "Never quote Poe; instead, echo the archetype of a knowing raven."
          ].join("\n"),
          developer: [
            "Constraints:",
            "- 6–10 short lines.",
            "- Refrain varies subtly (e.g., 'Nevermore', '…never more', 'No more').",
            "- Occasional internal rhyme acceptable."
          ].join("\n"),
          user_template: "Offer a raven's counsel about {topic}.",
          variables: [
            { key: "topic", label: "Topic", type: "string", default: "unquiet memory" }
          ],
          examples: [
            { input: "Offer a raven's counsel about unquiet memory." }
          ]
        }
      ]
    }
    fs.writeFileSync(DATA_PATH, JSON.stringify(seed, null, 2), 'utf-8')
  }
}

export function readDB(): PromptDB {
  ensureStore()
  const raw = fs.readFileSync(DATA_PATH, 'utf-8')
  return JSON.parse(raw)
}

export function writeDB(db: PromptDB){
  ensureStore()
  fs.writeFileSync(DATA_PATH, JSON.stringify(db, null, 2), 'utf-8')
}
